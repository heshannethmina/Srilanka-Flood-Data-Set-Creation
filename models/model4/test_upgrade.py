"""Checks for v2 sequence learning, projected reads and acquisition alignment."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model4.test_workflow import source, cfg
from model4.workflow import prepare, arrays_at, objective, build_model, run
from model4.data_audit import causal_frame_map, audit_index, read_training_panel


def test_sequence_order_and_trainable_branch(source,cfg):
    cfg=replace(cfg,revision='v2')
    data=prepare(*source,cfg)
    idx=np.argwhere(data['masks']['train'])[:12]
    x,s,state,y=arrays_at(data,idx,cfg,torch.device('cpu'))
    torch.manual_seed(43)
    model=build_model('model4',data,cfg).eval()
    summary=build_model('model4_summary',data,cfg).eval()
    changed=x.clone()
    changed[:,[3,5]]=changed[:,[5,3]]
    # Same summary statistics, different temporal order in the middle days.
    torch.testing.assert_close(summary(x,s,state),summary(changed,s,state),rtol=0,atol=1e-7)
    assert not torch.allclose(model.encode(x,s),model.encode(changed,s),rtol=0,atol=1e-8)
    p=model(x,s,state)
    assert p.shape==(len(x),cfg.members,4)
    assert torch.all(p[...,1:] >= 0)
    assert torch.all(p[...,0]<=p[...,1]) and torch.all(p[...,1]<=p[...,2])
    assert torch.all(p[state==1,:,3]==0)
    objective(p,y).backward()
    for param in [model.sequence.pos,model.sequence.embed.weight,model.sequence_gate.weight]:
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum()>0


def test_projected_parquet_read_preserves_inputs(source,tmp_path):
    frame,_=source
    frame=frame.copy(); frame['unused_large_text']='ignored payload'
    path=tmp_path/'flood_dataset.parquet'; frame.to_parquet(path,index=False)
    actual,audit=read_training_panel(path)
    assert 'unused_large_text' not in actual
    assert audit['rows']==len(frame)
    assert audit['total_columns']==len(frame.columns)
    np.testing.assert_array_equal(actual.discharge,frame.discharge.astype('float32'))


def test_frame_alignment_uses_acquisition_calendar_age_and_lag():
    idx=pd.DataFrame(dict(site_id=['A','A','A'],
        actual_date=['2021-01-06','2020-12-30','2021-01-01'],
        date=['2020-12-29']*3))
    dates=np.array(['2021-01-01','2021-01-02','2021-01-05','2021-01-06','2021-01-07','2021-02-01'],dtype='datetime64[D]')
    result=causal_frame_map(idx,['A','B'],dates,max_age=5,availability_lag=1)
    assert result['frame'][:,0].tolist()==[1,2,2,2,0,-1]
    assert result['age'][:,0].tolist()==[2,1,4,5,1,0]
    assert (result['frame'][:,1]==-1).all()
    # A future frame appended to the index cannot alter previous forecasts.
    changed=pd.concat([idx,pd.DataFrame(dict(site_id=['A'],actual_date=['2021-05-01'],date=['2020-01-01']))],ignore_index=True)
    np.testing.assert_array_equal(result['frame'],causal_frame_map(changed,['A','B'],dates,max_age=5)['frame'])


def test_image_audit_counts_gaps_and_label_conflicts(source,cfg,tmp_path):
    data=prepare(*source,cfg)
    index=pd.DataFrame(dict(site_id=['A','A','A'],date=['2021-01-01','2021-01-05','2021-01-10'],label=[0,0,0]))
    day=np.flatnonzero(data['dates']==np.datetime64('2021-01-01'))[0]
    index.loc[0,'label']=1-data['state'][day,0]
    path=tmp_path/'image_dataset.csv'; index.to_csv(path,index=False)
    report=audit_index(path,data)
    assert report['sites'][0]['median_gap_days']==4.5
    assert report['sites'][0]['max_gap_days']==5
    assert report['label_check']['disagreements']>=1
    assert report['panel_nodes_with_frames']==1
    assert report['coverage_by_split']['train']==0


def test_v2_end_to_end_and_resume(source,cfg,tmp_path,monkeypatch):
    import model4.data_audit as audit
    monkeypatch.setattr(audit,'audit_images',lambda data:dict(found=False,indices=[]))
    frame,nodes=source
    root=tmp_path/'data'; root.mkdir()
    frame.to_parquet(root/'flood_dataset.parquet',index=False)
    nodes.to_csv(root/'nodes.csv',index=False)
    out=tmp_path/'model4_v2'; cfg=replace(cfg,revision='v2')
    run(root,out,cfg)
    assert json.loads((out/'status.json').read_text())['complete']
    summary=pd.read_csv(out/'summary.csv')
    assert set(summary.model)=={'model4','model4_summary','model2_control','lightgbm','persistence','discharge_pctl'}
    intervals=json.loads((out/'paired_comparisons.json').read_text())
    assert 'model4_summary' in intervals
    metrics=json.loads((out/'metrics.json').read_text())
    for name in ['model4','model4_summary','model2_control']:
        assert metrics[name]['n_seeds']==len(cfg.seeds)
        assert np.isfinite(metrics[name]['heads']['flood_1d']['raw_brier'])
    assert (out/'image_audit.json').exists()
    checkpoint=out/'model4_seed0.pt'; before=checkpoint.stat().st_mtime_ns
    run(root,out,cfg)
    assert checkpoint.stat().st_mtime_ns==before


def test_v2_cli_defaults(tmp_path,monkeypatch):
    import kaggle_run as runner
    (tmp_path/'nodes.csv').write_text('node_id\nA\n')
    monkeypatch.setattr(sys,'argv',['kaggle_run.py','--stage','model4_v2'])
    monkeypatch.setattr(runner,'require_kaggle',lambda:None)
    monkeypatch.setattr(runner,'report_env',lambda:None)
    monkeypatch.setattr(runner,'find_input',lambda *args:str(tmp_path))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    calls=[]
    monkeypatch.setattr(runner,'stage_model4',lambda root,epochs,seeds,budget,batch,overrides,revision:calls.append((epochs,seeds,budget,batch,revision)))
    runner.main()
    assert calls==[(80,3,8.,512,'v2')]
