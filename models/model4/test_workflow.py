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


def test_notebook_uses_existing_runner_and_compiles():
    root=Path(__file__).resolve().parents[2]
    nb=json.loads((root/'notebooks/tfstgnn_kaggle.ipynb').read_text(encoding='utf-8'))
    code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    tree=ast.parse('\n\n'.join(code))
    assert not any(isinstance(n,ast.Name) and n.id=='__file__' for n in ast.walk(tree))
    assert "'git', 'clone'" in code[0]
    assert "'--stage', 'model4'" in code[2]
    standalone=(root/'notebooks/model4_kaggle.py').read_text(encoding='utf-8')
    assert ast.dump(ast.parse(standalone))==ast.dump(tree)
    import nbformat
    nbformat.validate(nb)


@pytest.mark.parametrize('stage,expected',[
    ('model4',(80,3,8.,1024)),
    ('sar',(60,5,256,8)),
])
def test_kaggle_cli_dispatch_preserves_stage_defaults(stage,expected,tmp_path,monkeypatch):
    import kaggle_run as runner
    (tmp_path/'nodes.csv').write_text('node_id\nA\n')
    monkeypatch.setattr(sys,'argv',['kaggle_run.py','--stage',stage])
    monkeypatch.setattr(runner,'require_kaggle',lambda:None)
    monkeypatch.setattr(runner,'report_env',lambda:None)
    monkeypatch.setattr(runner,'find_input',lambda *args:str(tmp_path))
    monkeypatch.setattr(runner,'summarise',lambda:None)
    monkeypatch.setattr(runner,'RUNS',str(tmp_path/'runs'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    calls=[]
    monkeypatch.setattr(runner,'stage_model4',lambda root,epochs,seeds,budget,batch,overrides:calls.append((epochs,seeds,budget,batch)))
    monkeypatch.setattr(runner,'stage_sar',lambda root,sar,epochs,seeds,px,batch:calls.append((epochs,seeds,px,batch)))
    runner.main()
    assert calls==[expected]


@pytest.mark.parametrize('fail',[False,True])
def test_model4_runner_packages_outputs_even_on_failure(tmp_path,monkeypatch,fail):
    import kaggle_run as runner
    from types import SimpleNamespace
    import zipfile
    monkeypatch.setattr(runner,'RUNS',str(tmp_path/'runs'))
    source=tmp_path/'models'; source.mkdir()
    (source/'example.py').write_text('pass\n')
    monkeypatch.setattr(runner,'HERE',str(source))
    monkeypatch.setattr(runner.glob,'glob',lambda *args,**kwargs:[])
    monkeypatch.setattr(runner.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='test-commit\n'))
    def fake_run(root,out,cfg):
        assert cfg.epochs==80 and cfg.seeds==(0,1,2) and cfg.budget_hours==7.5
        (out/'checkpoint-marker.txt').write_text('saved')
        if fail:
            raise RuntimeError('simulated training failure')
        (out/'status.json').write_text(json.dumps({'complete':False,'pending':['seed 1']}))
    monkeypatch.setattr(workflow,'run',fake_run)
    if fail:
        with pytest.raises(RuntimeError,match='simulated training failure'):
            runner.stage_model4(str(tmp_path),80,3,8.,1024)
    else:
        assert not runner.stage_model4(str(tmp_path),80,3,8.,1024)['complete']
    with zipfile.ZipFile(tmp_path/'runs.zip') as archive:
        assert 'runs/model4/checkpoint-marker.txt' in archive.namelist()
        assert 'runs/model4/source/models/example.py' in archive.namelist()
        if fail:
            assert 'simulated training failure' in archive.read('runs/model4/failure.txt').decode()


def test_real_grad_scaler_skips_overflow_before_clipping():
    model=torch.nn.Linear(1,1,bias=False)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
    scaler=torch.amp.GradScaler('cpu',init_scale=128.)
    scaler.scale(model(torch.ones(1,1)).sum()).backward()
    model.weight.grad.fill_(float('inf'))
    before=model.weight.detach().clone()
    assert not workflow.safe_optimizer_step(model,optimizer,scaler,2.)
    torch.testing.assert_close(model.weight,before,rtol=0,atol=0)
    assert scaler.get_scale()==64. and not optimizer.state
    scaler.scale(model(torch.ones(1,1)).sum()).backward()
    assert workflow.safe_optimizer_step(model,optimizer,scaler,2.)
    assert not torch.equal(model.weight,before) and optimizer.state


@pytest.mark.parametrize('bad_attempts',[1,3,4])
def test_batch_retries_are_bounded_and_fallback_to_fp32(bad_attempts):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.tensor(0.))
            self._amp_enabled=True
            self.calls=0
            self.weight.register_hook(self.inject_overflow)
        def inject_overflow(self,gradient):
            return torch.full_like(gradient,float('inf')) if self.calls<=bad_attempts else gradient
        def forward(self,x,s,state):
            self.calls+=1
            return self.weight.sigmoid().expand(len(x),1,4)
    model=Tiny()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scaler=torch.amp.GradScaler('cpu',init_scale=128.)
    x=torch.ones(2,2,1); s=torch.ones(2,1); state=torch.zeros(2); y=torch.zeros(2,4)
    if bad_attempts==4:
        with pytest.raises(FloatingPointError,match='full precision'):
            workflow.train_batch(model,optimizer,scaler,x,s,state,y,2.)
        assert model.calls==4 and model.weight.item()==0. and not optimizer.state
    else:
        loss,scaler,retries=workflow.train_batch(model,optimizer,scaler,x,s,state,y,2.)
        assert np.isfinite(loss) and model.weight.item()<0.
        assert retries==bad_attempts and model.calls==bad_attempts+1
        assert int(optimizer.state[model.weight]['step'])==1
        assert model._amp_enabled==(bad_attempts<3)
        assert scaler.is_enabled()==model._amp_enabled


def test_legacy_checkpoint_resume_and_precision_mode(source,cfg,tmp_path):
    data=prepare(*source,cfg)
    model,_=fit_seed('model4',0,data,cfg,tmp_path,time.monotonic()+120,torch.device('cpu'))
    path=tmp_path/'model4_seed0.pt'
    saved=torch.load(path,weights_only=False)
    saved.pop('amp_enabled')  # exact pre-fix checkpoint field set
    torch.save(saved,path)
    restored,_=fit_seed('model4',0,data,cfg,tmp_path,time.monotonic()+120,torch.device('cpu'))
    assert not restored._amp_enabled
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
    saved['state'][next(k for k,v in saved['state'].items() if v.is_floating_point())].fill_(float('nan'))
    torch.save(saved,path)
    with pytest.raises(FloatingPointError,match='refusing resume'):
        fit_seed('model4',0,data,cfg,tmp_path,time.monotonic()+120,torch.device('cpu'))


@pytest.mark.parametrize('changed',['none','config','data','other_code','unknown_version','predictions','complete'])
def test_only_known_unfinished_run_can_migrate(tmp_path,changed):
    current=dict(config={'seeds':(0,1,2),'epochs':80},data_sha256='data',nodes_sha256='nodes',
                 source_sha256={'model4/workflow.py':'new','model2/model.py':'unchanged'})
    previous=json.loads(json.dumps(current))
    previous['source_sha256']['model4/workflow.py']=workflow.PRE_AMP_FIX_SHA256
    previous['signature']='previous'
    if changed=='config': previous['config']['epochs']=60
    if changed=='data': previous['data_sha256']='different'
    if changed=='other_code': previous['source_sha256']['model2/model.py']='different'
    if changed=='unknown_version': previous['source_sha256']['model4/workflow.py']='unrecognised'
    if changed=='predictions': (tmp_path/'model2_control_seed0_predictions.npz').touch()
    if changed=='complete': (tmp_path/'status.json').write_text('{"complete": true}')
    (tmp_path/'manifest.json').write_text(json.dumps(previous))
    if changed!='none':
        with pytest.raises(ValueError,match='different dataset/code/config'):
            workflow.verify_resume_manifest(tmp_path,current,'new-signature')
        assert not (tmp_path/'manifest_before_amp_fix.json').exists()
    else:
        migrations=workflow.verify_resume_manifest(tmp_path,current,'new-signature')
        assert len(migrations)==1 and migrations[0]['from_signature']=='previous'
        assert json.loads((tmp_path/'manifest_before_amp_fix.json').read_text())==previous
