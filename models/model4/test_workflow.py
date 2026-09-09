"""Regression tests for scientific correctness and the packaged workflow.

Run: python -m pytest models/model4/test_workflow.py -q
"""
import ast
import json
from pathlib import Path
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model4.workflow import (Config,prepare,HydroTEM,Model2Control,objective,
    arrays_at,fit_seed,predict,ap,choose_threshold,calibrate,fit_calibrator,
    paired_bootstrap,run,event_report)
from floodlib.schema import DYNAMIC_FEATURES
import model4.workflow as workflow


@pytest.fixture
def source():
    dates = pd.date_range('2017-10-01','2021-03-31')
    frames=[]
    for i,n in enumerate(['A','B']):
        t=np.arange(len(dates))
        q=(t%20+1)*(i+1.)
        df=pd.DataFrame({name:np.sin(t/13)+2 for name in DYNAMIC_FEATURES})
        df['date']=dates; df['node_id']=n
        df['discharge']=q
        df['discharge_pctl']=(t%20+1)/20
        df['valid_sample']=1
        frames.append(df)
    nodes=pd.DataFrame(dict(node_id=['A','B'],elevation_m=[10.,20.],
                            zone=['wet','dry'],position=['upstream','outlet'],basin=['one','two']))
    return pd.concat(frames,ignore_index=True),nodes


@pytest.fixture
def cfg():
    torch.set_num_threads(2)
    return Config(width=16,embedding=3,members=2,bins=5,epochs=1,batch_size=256,
                  seeds=(0,),bootstrap_draws=3)


def test_future_changes_never_fit_preprocessing(source,cfg):
    df,nodes=source
    a=prepare(df,nodes,cfg)
    changed=df.copy()
    changed.loc[changed.date>='2018-01-01',DYNAMIC_FEATURES]=12345.
    b=prepare(changed,nodes,cfg)
    for key in ['median','mean','std','knots','threshold','static']:
        np.testing.assert_array_equal(a['norm'][key],b['norm'][key])
    assert not a['masks']['train'][(a['dates']>np.datetime64('2017-12-28'))].any()
    for date in ['2019-12-29','2019-12-30','2019-12-31','2020-12-29','2020-12-30','2020-12-31']:
        d=np.flatnonzero(a['dates']==np.datetime64(date))[0]
        assert not any(m[d].any() for m in a['masks'].values())


def test_target_windows_and_missing_days(source,cfg):
    df,nodes=source
    # Remove an observation, retain daily index: don't shift across missing days.
    df=df[~((df.node_id=='A') & (df.date==pd.Timestamp('2021-02-04')))]
    a=prepare(df,nodes,cfg)
    d=np.flatnonzero(a['dates']==np.datetime64('2021-02-04'))[0]
    assert not a['masks']['test'][d-3:d+1,0].any()
    for day,node in np.argwhere(a['masks']['test'])[::13]:
        st=a['state']
        for h in range(3):
            assert a['y'][day,node,h]==max(st[day+1:day+h+2,node])
        assert a['y'][day,node,3]==a['y'][day,node,0]*(1-st[day,node])


def test_constraints_and_all_members_receive_gradients(source,cfg):
    data=prepare(*source,cfg)
    idx=np.argwhere(data['masks']['train'])[:40]
    x,s,state,y=arrays_at(data,idx,cfg,torch.device('cpu'))
    model=HydroTEM(data,cfg)
    p=model(x,s,state)
    assert (p[...,0]<=p[...,1]).all() and (p[...,1]<=p[...,2]).all()
    assert (p[state==1,:,3]==0).all()
    torch.testing.assert_close(p[state==0,:,3],p[state==0,:,0])
    objective(p,y).backward()
    assert torch.isfinite(model.head.r.grad).all()
    assert (model.head.r.grad.abs().sum(-1)>0).all()
    # Existing Model 2 accepts the exact same input contract and trains.
    control=Model2Control(data,cfg)
    q=control(x[:3],s[:3],state[:3]); objective(q,y[:3]).backward()
    assert q.shape==(3,1,4)


def test_ties_thresholds_and_ordering():
    y=np.array([0,1,0,1]); p=np.array([.5,.5,.5,.5])
    assert ap(y,p)==.5
    assert ap(y[::-1],p)==.5
    assert choose_threshold(y,p,max_far=.1)>1
    ordered=np.array([[.01,.1,.4],[.2,.5,.6]])
    assert (np.diff(calibrate(ordered,[2.,-1]),axis=1)>=0).all()
    boot=paired_bootstrap(y,p,p,np.array(['2021-01-01','2021-01-02','2021-03-01','2021-03-02'],dtype='datetime64[D]'),10)
    assert boot['delta_ap']==0 and boot['ci95']==[0,0]


def test_checkpoint_predict_roundtrip(source,cfg,tmp_path):
    data=prepare(*source,cfg)
    model,history=fit_seed('model4',0,data,cfg,tmp_path,time.monotonic()+120,torch.device('cpu'))
    idx=np.argwhere(data['masks']['test'])[:40]
    p=predict(model,data,idx,cfg,torch.device('cpu'))
    restored,h=fit_seed('model4',0,data,cfg,tmp_path,time.monotonic()+120,torch.device('cpu'))
    np.testing.assert_array_equal(p,predict(restored,data,idx,cfg,torch.device('cpu')))
    assert len(history)==len(h)==1


def test_interrupted_epoch_resume_is_equivalent(source,cfg,tmp_path,monkeypatch):
    cfg=replace(cfg,epochs=2)
    data=prepare(*source,cfg)
    a=tmp_path/'a'; b=tmp_path/'b'; a.mkdir(); b.mkdir()
    expected,_=fit_seed('model4',0,data,cfg,a,time.monotonic()+120,torch.device('cpu'))
    original=workflow.torch_save
    def interrupted(path,payload):
        original(path,payload)
        raise workflow.BudgetExpired('test interruption after saved epoch')
    monkeypatch.setattr(workflow,'torch_save',interrupted)
    with pytest.raises(workflow.BudgetExpired):
        fit_seed('model4',0,data,cfg,b,time.monotonic()+120,torch.device('cpu'))
    monkeypatch.setattr(workflow,'torch_save',original)
    restored,_=fit_seed('model4',0,data,cfg,b,time.monotonic()+120,torch.device('cpu'))
    for key,value in expected.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)


def test_budget_outputs_are_explicitly_partial(source,cfg,tmp_path):
    df,nodes=source
    df.to_parquet(tmp_path/'flood_dataset.parquet'); nodes.to_csv(tmp_path/'nodes.csv',index=False)
    out=tmp_path/'out'
    run(tmp_path,out,replace(cfg,budget_hours=0))
    status=json.loads((out/'status.json').read_text())
    assert not status['complete'] and len(status['pending'])==3
    assert not (out/'summary.csv').exists()


def test_end_to_end_artifacts(source,cfg,tmp_path):
    df,nodes=source
    root=tmp_path/'data'; root.mkdir()
    df.to_parquet(root/'flood_dataset.parquet'); nodes.to_csv(root/'nodes.csv',index=False)
    out=tmp_path/'out'
    run(root,out,cfg)
    assert json.loads((out/'status.json').read_text())['complete']
    summary=pd.read_csv(out/'summary.csv')
    assert len(summary)==6*4
    with np.load(out/'model4_predictions.npz') as pp:
        assert np.all(np.diff(pp['p'][:,:3],axis=1)>=-1e-7)
    assert (out/'evaluation.png').stat().st_size>1000
    # Completed rerun uses cached predictions/checkpoints, not new training.
    checkpoint=out/'model4_seed0.pt'; before=checkpoint.stat().st_mtime_ns
    run(root,out,cfg)
    assert checkpoint.stat().st_mtime_ns==before


def test_notebook_bundle_exact_and_compiles():
    root=Path(__file__).resolve().parents[2]
    nb=json.loads((root/'notebooks/tfstgnn_kaggle.ipynb').read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            src=''.join(cell['source']); tree=ast.parse(src)
            for stmt in tree.body:
                if isinstance(stmt,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in stmt.targets):
                    for name,code in ast.literal_eval(stmt.value).items():
                        assert code==(root/name).read_text(encoding='utf-8')
                        compile(code,name,'exec')
    import nbformat
    nbformat.validate(nb)
