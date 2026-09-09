"""Model 4 experiment, launched by kaggle_run.py --stage model4.

Research candidate, not a claim of improvement. See docs/MODEL4.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve)
from scipy.optimize import minimize

from floodlib.schema import DYNAMIC_FEATURES, LOG1P_FEATURES, ZONES, POSITIONS
from model2.config import MMFConfig
from model2.model import MMFNet


@dataclass
class Config:
    lookback: int = 14
    width: int = 128
    members: int = 4
    bins: int = 24
    embedding: int = 8
    dropout: float = 0.15
    epochs: int = 80
    patience: int = 12
    batch_size: int = 1024
    lr: float = 0.0005
    weight_decay: float = 0.0001
    grad_clip: float = 2.0
    seeds: tuple = (0, 1, 2)
    budget_hours: float = 7.5
    bootstrap_draws: int = 500
    max_far: float = 0.231


def json_save(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        return x
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def torch_save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def prepare(df, nodes, cfg):
    """Dense daily panel; rebuild full-horizon targets and purge boundaries.

    All normalisers/knots use training origins only. Historical context may
    cross a split boundary; the future labels may not. No backward filling.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df.date).dt.normalize()
    df = df.loc[df.date.between('2003-01-01', '2024-12-31')]
    if df.empty or df.duplicated(['node_id', 'date']).any():
        raise ValueError('Panel is empty or contains duplicate node/day keys.')
    required = ['discharge', 'discharge_pctl', 'valid_sample'] + DYNAMIC_FEATURES
    absent = sorted(set(required) - set(df))
    if absent:
        raise ValueError(f'Incomplete tabular dataset: missing {absent}')
    ids = sorted(df.node_id.unique())
    dates = pd.date_range(df.date.min(), df.date.max(), freq='D')
    grid = pd.MultiIndex.from_product([dates, ids], names=['date', 'node_id'])
    g = df.set_index(['date', 'node_id']).reindex(grid)
    T, N = len(dates), len(ids)
    raw = g[DYNAMIC_FEATURES].to_numpy(np.float32, copy=True).reshape(T, N, -1)
    raw[~np.isfinite(raw)] = np.nan
    q = raw[..., DYNAMIC_FEATURES.index('discharge')]
    train_dates = dates <= '2017-12-31'
    threshold = np.nanquantile(q[train_dates], .98, axis=0).astype('float32')
    if 'thr_high' in g:
        # The archived threshold defines the task; verify it against train Q.
        stored = g['thr_high'].to_numpy(np.float32).reshape(T, N)
        archived = np.nanmedian(stored[train_dates], axis=0)
        if not np.allclose(archived, threshold, rtol=1e-3, atol=1e-4):
            raise ValueError('Archived high thresholds differ from 2003-2017 Q98; inspect dataset version.')
        threshold = archived
    if not np.isfinite(threshold).all() or np.any(threshold <= 0):
        raise ValueError('Missing or nonpositive per-node flood threshold.')
    state = np.where(np.isfinite(q), (q >= threshold).astype(float), np.nan)
    future = np.full((T, N, 3), np.nan, np.float32)
    for h in range(1, 4):
        future[:-h, :, h-1] = state[h:]
    y = np.maximum.accumulate(future, axis=-1)
    y = np.concatenate([y, (y[..., :1] * (1-state[..., None]))], axis=-1)
    # Separate model selection (2018-19), calibration (2020), and test (2021-24).
    period = np.select([dates <= '2017-12-31', dates <= '2019-12-31',
                        dates <= '2020-12-31'], [0, 1, 2], default=3)
    valid = (g.valid_sample.fillna(0).to_numpy().reshape(T, N) > 0)
    valid &= np.isfinite(future).all(-1) & np.isfinite(state)
    for h in range(1, 4):
        same = np.zeros(T, bool)
        same[:-h] = period[:-h] == period[h:]
        valid &= same[:, None]
    valid[:cfg.lookback-1] = False
    masks = {name: valid & (period[:, None] == i)
             for i, name in enumerate(['train', 'select', 'calibrate', 'test'])}
    if any(not m.any() for m in masks.values()):
        raise ValueError('All four chronological periods require usable samples.')
    # A directly interpretable causal feature: distance from flood threshold.
    ratio = np.log(np.maximum(q, 1e-6) / threshold)[..., None]
    raw = np.concatenate([raw, ratio], axis=-1)
    feature_names = DYNAMIC_FEATURES + ['log_q_over_train_q98']
    for i, name in enumerate(feature_names):
        if name in LOG1P_FEATURES:
            raw[..., i] = np.sign(raw[..., i]) * np.log1p(np.abs(raw[..., i]))
    tr = raw[masks['train']]
    median = np.nanmedian(tr, axis=0)
    median = np.where(np.isfinite(median), median, 0)
    missing = ~np.isfinite(raw)
    filled = np.where(missing, median, raw)
    mean = filled[masks['train']].mean(0)
    std = filled[masks['train']].std(0)
    std = np.where(std > 1e-6, std, 1)
    numeric = np.clip((filled-mean)/std, -12, 12).astype('float32')
    phase = 2*np.pi*(dates.dayofyear.to_numpy()-1)/365.25
    season = np.broadcast_to(np.stack([np.sin(phase), np.cos(phase)], -1)[:, None], (T,N,2))
    x = np.concatenate([numeric, missing.astype('float32'), season,
                        np.nan_to_num(state)[...,None]], -1).astype('float32')
    features = feature_names + [f'{f}__missing' for f in feature_names] + ['doy_sin','doy_cos','current_flood_state']
    nd = nodes.set_index('node_id').reindex(ids)
    if nd[['elevation_m', 'zone', 'position', 'basin']].isna().any().any():
        raise ValueError('nodes.csv must provide terrain metadata for every panel node.')
    drainage = np.log1p(np.nanmean(q[train_dates], axis=0))
    cont = np.column_stack([nd.elevation_m.to_numpy(float), drainage])
    sm, ss = cont.mean(0), cont.std(0).clip(1e-6)
    s = np.column_stack([(cont-sm)/ss, *[(nd.zone == z).to_numpy(float) for z in ZONES],
                         *[(nd.position == p).to_numpy(float) for p in POSITIONS]]).astype('float32')
    knots = np.quantile(numeric[masks['train']], np.linspace(0,1,cfg.bins+1), axis=0).T
    mismatch = {}
    for h, col in enumerate(['target_flood_1d','target_flood_2d','target_flood_3d','target_onset_1d']):
        if col in g:
            old = g[col].to_numpy(float).reshape(T,N)
            check = valid & np.isfinite(old)
            mismatch[col] = int(np.sum(old[check] != y[...,h][check]))
    audit = {'shape': [T,N,x.shape[-1]], 'features': features, 'numeric_features': feature_names,
             'nodes': ids, 'thresholds_q98': threshold,
             'split_counts': {k:int(v.sum()) for k,v in masks.items()},
             'positive_rates': {k:y[v].mean(0) for k,v in masks.items()},
             'target_disagreements_on_usable_rows': mismatch,
             'inherited_limitation': 'Source weather/soil interpolation and backfill cannot be undone from processed parquet. Reanalysis retrospective experiment, not operational validation.'}
    norm = dict(median=median, mean=mean, std=std, static_mean=sm, static_std=ss,
                knots=knots.astype('float32'), threshold=threshold, features=np.array(features),
                node_ids=np.array(ids), static=s)
    return dict(x=x, s=s, y=np.nan_to_num(y).astype('float32'), state=state,
                q=q, dates=dates.to_numpy('datetime64[D]'), ids=ids, basins=nd.basin.to_numpy(),
                masks=masks, knots=knots.astype('float32'), numeric_dim=len(feature_names),
                audit=audit, norm=norm)


class PiecewiseEmbedding(nn.Module):
    """Train-quantile piecewise-linear numeric embeddings (Gorishniy et al.)."""
    def __init__(self, knots, dim):
        super().__init__()
        self.register_buffer('left', torch.as_tensor(knots[:, :-1]).float())
        self.register_buffer('width', torch.as_tensor(np.diff(knots)).float().clamp_min(1e-5))
        self.weight = nn.Parameter(torch.randn(*self.left.shape, dim)*0.03)
        self.bias = nn.Parameter(torch.zeros(self.left.shape[0], dim))

    def forward(self, x):
        basis = ((x.unsqueeze(-1)-self.left)/self.width).clamp(0,1)
        return F.gelu(torch.einsum('bfi,fie->bfe', basis, self.weight)+self.bias).flatten(1)


class EnsembleLinear(nn.Module):
    """Shared matrix and member-specific rank-one input/output modulation.

    Inspired by BatchEnsemble/TabM; this custom temporal network is not the
    authors' TabM implementation. Each member receives its own BCE loss.
    """
    def __init__(self, din, dout, k):
        super().__init__()
        self.linear = nn.Linear(din, dout, bias=False)
        self.r = nn.Parameter(torch.empty(k,din).bernoulli_(.5)*2-1)
        self.s = nn.Parameter(torch.ones(k,dout))
        self.bias = nn.Parameter(torch.zeros(k,dout))

    def forward(self, x):
        return self.linear(x*self.r)*self.s+self.bias


class HydroTEM(nn.Module):
    def __init__(self, data, cfg, temporal=True):
        super().__init__()
        self.k, self.numeric_dim, self.temporal = cfg.members, data['numeric_dim'], temporal
        n = data['x'].shape[-1]
        self.embed = PiecewiseEmbedding(data['knots'], cfg.embedding)
        din = self.numeric_dim*cfg.embedding + n + data['s'].shape[-1]
        self.current = nn.Sequential(nn.Linear(din,cfg.width), nn.GELU(), nn.LayerNorm(cfg.width))
        if temporal:
            self.history = nn.Sequential(nn.Linear(n*4,cfg.width), nn.GELU(),
                                         nn.Dropout(cfg.dropout), nn.Linear(cfg.width,cfg.width))
            # Gate starts almost closed: protect the strong current-day branch.
            self.gate = nn.Parameter(torch.full((cfg.width,), -2.0))
        self.layers = nn.ModuleList([EnsembleLinear(cfg.width,cfg.width,self.k) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.width) for _ in range(3)])
        self.drop = nn.Dropout(cfg.dropout)
        self.head = EnsembleLinear(cfg.width,4,self.k)
        with torch.no_grad():
            # separate dry/wet first-day hazard, then conditional day-2/day-3 hazards
            self.head.bias[:] = torch.tensor([-4., 1., -4., -4.])

    def forward(self, x, s, state):
        cur = x[:,-1]
        h = self.current(torch.cat([self.embed(cur[:,:self.numeric_dim]),cur,s],-1))
        if self.temporal:
            # Observed lags and changes, no target-time weather/discharge.
            history = torch.cat([x[:,-2], x[:,-4:].mean(1), x.mean(1), cur-x[:,0]],-1)
            h = h + self.gate.sigmoid()*self.history(history)
        h = h[:,None].expand(-1,self.k,-1)
        for layer, norm in zip(self.layers,self.norms):
            h = norm(h+self.drop(F.gelu(layer(h))))
        z = self.head(h).float()
        first = z[...,0]*(1-state[:,None]) + z[...,1]*state[:,None]
        hazards = torch.stack([first,z[...,2],z[...,3]],-1)
        # Stable cumulative event probabilities: P(any flood within h).
        p = -torch.expm1(F.logsigmoid(-hazards).cumsum(-1))
        return torch.cat([p,p[...,:1]*(1-state[:,None,None])],-1)


class Model2Control(nn.Module):
    def __init__(self, data, cfg):
        super().__init__()
        self.net = MMFNet(MMFConfig(n_dynamic=data['x'].shape[-1], n_static=data['s'].shape[-1],
                                    lookback=cfg.lookback))

    def forward(self, x, s, state):
        # Existing network uses [snapshot, time, node, feature]; nodes are
        # independent with graph disabled, so one snapshot is equivalent.
        out = self.net(x.permute(1,0,2).unsqueeze(0), s)
        return out['logits'].squeeze(0).sigmoid().unsqueeze(1)


def build_model(name, data, cfg):
    return Model2Control(data,cfg) if name == 'model2_control' else HydroTEM(data,cfg, name != 'model4_current')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def arrays_at(data, indices, cfg, device):
    d,n = indices[:,0],indices[:,1]
    window = d[:,None]-np.arange(cfg.lookback-1,-1,-1)[None]
    x = torch.as_tensor(data['x'][window,n[:,None]],device=device)
    s = torch.as_tensor(data['s'][n],device=device)
    state = torch.as_tensor(data['state'][d,n],device=device,dtype=torch.float32)
    y = torch.as_tensor(data['y'][d,n],device=device)
    return x,s,state,y


def objective(p,y):
    # No focal weighting or oversampling. Each member is supervised separately.
    losses = F.binary_cross_entropy(p.float().clamp(1e-6,1-1e-6),
                                   y[:,None].expand_as(p),reduction='none')
    return (losses*losses.new_tensor([1.,.3,.3,.5])).sum(-1).mean()


@torch.no_grad()
def predict(model,data,idx,cfg,device):
    model.eval()
    result = []
    for start in range(0,len(idx),cfg.batch_size):
        x,s,state,_ = arrays_at(data,idx[start:start+cfg.batch_size],cfg,device)
        with torch.autocast(device_type=device.type, enabled=device.type=='cuda'):
            p = model(x,s,state)
        result.append(p.float().mean(1).cpu().numpy())
    return np.concatenate(result)


def ap(y,p):
    return float(average_precision_score(y,p)) if np.unique(y).size == 2 else float('nan')


class BudgetExpired(Exception):
    pass


def fit_seed(name,seed,data,cfg,out,deadline,device):
    seed_all(seed)
    checkpoint = out/f'{name}_seed{seed}.pt'
    model = build_model(name,data,cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start,best,bad,history,best_state = 0,-np.inf,0,[],None
    if checkpoint.exists():
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['complete']:
            model.load_state_dict(saved['best_state'])
            return model, saved['history']
        model.load_state_dict(saved['state']); opt.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        start,best,bad,history,best_state = (saved[k] for k in ['epoch','best','bad','history','best_state'])
        torch.set_rng_state(saved['rng'])
        np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
        if device.type=='cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    train_idx = np.argwhere(data['masks']['train'])
    val_idx = np.argwhere(data['masks']['select'])
    target = data['y'][data['masks']['select']]
    for epoch in range(start,cfg.epochs):
        tic = time.monotonic()
        # Reserve enough time for validation and a checkpoint before deadline.
        if tic+max(60, history[-1]['seconds']*1.3 if history else 60) >= deadline:
            raise BudgetExpired(name)
        lr = cfg.lr*min(1.,(epoch+1)/5)*(.1+.9*.5*(1+math.cos(math.pi*epoch/cfg.epochs)))
        for group in opt.param_groups:
            group['lr']=lr
        model.train()
        shuffled = train_idx[np.random.permutation(len(train_idx))]
        total,seen = 0.,0
        for i in range(0,len(shuffled),cfg.batch_size):
            if time.monotonic() >= deadline-30:
                raise BudgetExpired(name)
            idx = shuffled[i:i+cfg.batch_size]
            x,s,state,y = arrays_at(data,idx,cfg,device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type=='cuda'):
                p = model(x,s,state)
            loss = objective(p,y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{name} seed {seed}: nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
            scaler.step(opt); scaler.update()
            total += float(loss.detach())*len(idx); seen += len(idx)
        v = predict(model,data,val_idx,cfg,device)
        score = ap(target[:,0],v[:,0])
        if not np.isfinite(score):
            raise ValueError('Selection period needs both positive and negative flood labels.')
        if score > best+1e-5:
            best,bad = score,0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad += 1
        history.append(dict(epoch=epoch+1,loss=total/seen,val_ap=score,
                            val_onset_ap=ap(target[:,3],v[:,3]),lr=lr,seconds=time.monotonic()-tic))
        complete = bad>=cfg.patience or epoch+1==cfg.epochs
        torch_save(checkpoint,dict(state=model.state_dict(),best_state=best_state,optimizer=opt.state_dict(),
                                  scaler=scaler.state_dict(),epoch=epoch+1,best=best,bad=bad,history=history,
                                  complete=complete,rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
                                  python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
        print(f'{name} seed={seed} epoch={epoch+1:02d} loss={total/seen:.5f} val_AP={score:.4f} best={best:.4f} ({time.monotonic()-tic:.0f}s)',flush=True)
        if complete:
            break
    model.load_state_dict(best_state)
    return model,history


def fit_calibrator(y,p):
    """Positive-slope logit scaling fitted exclusively on 2020.

    One common mapping across the flood horizons preserves their ordering.
    Onset is mapped separately for controls and derived exactly for Model 4.
    """
    if np.unique(y).size < 2:
        return [1.,0.]
    z = np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))
    def loss(ab):
        zz = ab[0]*z+ab[1]
        return np.mean(np.logaddexp(0,zz)-y*zz)
    fit = minimize(loss,[1.,0.],method='L-BFGS-B',bounds=[(.05,10),(-10,10)])
    return fit.x.tolist() if fit.success else [1.,0.]


def calibrate(p,ab):
    z = np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))
    return 1/(1+np.exp(-np.clip(ab[0]*z+ab[1],-50,50)))


def choose_threshold(y,p,max_far=None):
    precision,recall,threshold = precision_recall_curve(y,p)
    score = recall[:-1] if max_far is not None else 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    ok = np.ones(len(threshold),bool) if max_far is None else (1-precision[:-1]<=max_far)
    if not ok.any():
        return 1.000001  # Explicit no-feasible-alarm policy, never silently .5.
    score = np.where(ok,score,-np.inf)
    return float(threshold[np.argmax(score)])


def metrics(y,p,threshold):
    alarms = p>=threshold
    tp,fp,fn = int(np.sum(alarms & (y==1))),int(np.sum(alarms & (y==0))),int(np.sum(~alarms & (y==1)))
    ece = 0.
    for k in range(15):
        m = np.minimum((p*15).astype(int),14)==k
        if m.any():
            ece += m.mean()*abs(y[m].mean()-p[m].mean())
    return dict(pr_auc=ap(y,p),roc_auc=float(roc_auc_score(y,p)) if np.unique(y).size==2 else None,
                brier=float(np.mean((p-y)**2)),ece=float(ece),n=len(y),positive_rate=float(y.mean()),
                threshold=threshold,tp=tp,fp=fp,fn=fn,pod=tp/max(tp+fn,1),far=fp/max(tp+fp,1),
                csi=tp/max(tp+fp+fn,1),f1=2*tp/max(2*tp+fp+fn,1))


def event_report(data,idx,p,threshold,lead=3):
    """True full-panel starts, complete forecast windows, currently dry alarms.

    A 3-day merged episode separates events by at least three dry days. This
    shares the catalogue's <=2-day flood-to-flood gap convention.
    """
    lookup = {(int(d),int(n)):float(prob) for (d,n),prob in zip(idx,p)}
    detected,total,leads = 0,0,[]
    for n in range(len(data['ids'])):
        wet = np.flatnonzero(data['state'][:,n]==1)
        starts = wet[np.r_[True,np.diff(wet)>2]] if wet.size else []
        for d in starts:
            window = range(d-lead,d)
            if d<lead or not all((t,n) in lookup for t in window):
                continue
            total += 1
            hit = [d-t for t in window if data['state'][t,n]==0 and lookup[t,n]>=threshold]
            if hit:
                detected += 1; leads.append(max(hit))
    return dict(n_eligible_events=total,n_detected=detected,event_detection_rate=detected/max(total,1),
                mean_lead_days=float(np.mean(leads)) if leads else None,max_lead_days=lead)


def paired_bootstrap(y,a,b,dates,draws):
    # Resample contiguous 28-day blocks; keep ALL nodes in each time block.
    blocks = (dates-dates.min()).astype('timedelta64[D]').astype(int)//28
    unique = np.unique(blocks)
    rng = np.random.default_rng(712)
    diffs = []
    # Sample weights avoid copying or concatenating hundreds of thousands of rows.
    _,codes = np.unique(blocks,return_inverse=True)
    for _ in range(draws):
        counts = np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
        w = counts[codes]
        if np.sum(w*y)==0 or np.sum(w*(1-y))==0:
            continue
        diffs.append(average_precision_score(y,a,sample_weight=w)-average_precision_score(y,b,sample_weight=w))
    return dict(delta_ap=ap(y,a)-ap(y,b),ci95=np.quantile(diffs,[.025,.975]).tolist() if diffs else None,
                draws=len(diffs),block_days=28,spatial_unit='all nodes jointly',
                caveat='Conditional on fitted models; does not include training uncertainty or test-reuse selection bias.')


def baseline_predictions(data,cfg,out):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    import joblib
    indices = {k:np.argwhere(v) for k,v in data['masks'].items()}
    def features(idx):
        d,n = idx.T
        return np.concatenate([data['x'][d,n],data['s'][n],
                               data['x'][d-1,n],data['x'][d-3,n],data['x'][d,n]-data['x'][d-7,n]],-1)
    xx = {k:features(v) for k,v in indices.items()}
    yy = {k:data['y'][data['masks'][k]] for k in indices}
    predictions = {name:{k:[] for k in ['calibrate','test']} for name in ['lightgbm','persistence','discharge_pctl']}
    for h in range(4):
        tree = LGBMClassifier(n_estimators=1500,learning_rate=.025,num_leaves=31,min_child_samples=100,
                              metric='average_precision',
                              colsample_bytree=.9,reg_lambda=2.,n_jobs=4,random_state=0,verbosity=-1)
        tree.fit(xx['train'],yy['train'][:,h],eval_set=[(xx['select'],yy['select'][:,h])],
                 eval_metric='average_precision',callbacks=[early_stopping(75,first_metric_only=True),log_evaluation(0)])
        joblib.dump(tree,out/f'lightgbm_head{h}.joblib')
        for k in ['calibrate','test']:
            d,n = indices[k].T
            predictions['lightgbm'][k].append(tree.predict_proba(xx[k])[:,1])
            # Persistent state predicts no onset. This is the actual baseline.
            predictions['persistence'][k].append(data['state'][d,n] if h<3 else np.zeros(len(d)))
            pos = DYNAMIC_FEATURES.index('discharge_pctl')
            percentile = data['x'][d,n,pos]*data['norm']['std'][pos]+data['norm']['mean'][pos]
            predictions['discharge_pctl'][k].append(percentile if h<3 else percentile*(1-data['state'][d,n]))
    return {name:{k:np.stack(v,-1) for k,v in pp.items()} for name,pp in predictions.items()}


def evaluate_all(predictions,data,cfg,out,seed_metrics):
    ci = np.argwhere(data['masks']['calibrate']); ti = np.argwhere(data['masks']['test'])
    cy = data['y'][data['masks']['calibrate']]; ty = data['y'][data['masks']['test']]
    rows,report,calibrated = [],{},{}
    names = ['flood_1d','flood_2d','flood_3d','onset_1d']
    for name,pp in predictions.items():
        cp,tp = pp['calibrate'].copy(),pp['test'].copy()
        # Scaling is fit on calibration only. No test-tuned blend or threshold.
        ab = fit_calibrator(cy[:,:3].ravel(),cp[:,:3].ravel())
        cp[:,:3],tp[:,:3] = calibrate(cp[:,:3],ab),calibrate(tp[:,:3],ab)
        if name.startswith('model4'):
            cp[:,3]=cp[:,0]*(1-data['state'][tuple(ci.T)])
            tp[:,3]=tp[:,0]*(1-data['state'][tuple(ti.T)])
            onset_ab = None
        elif name=='persistence':
            cp[:,3]=0; tp[:,3]=0; onset_ab=None
        else:
            onset_ab = fit_calibrator(cy[:,3],cp[:,3])
            cp[:,3],tp[:,3]=calibrate(cp[:,3],onset_ab),calibrate(tp[:,3],onset_ab)
        calibrated[name]=tp
        seeds = seed_metrics.get(name,[])
        result = dict(calibration=dict(flood=ab,onset=onset_ab),heads={},seed_metrics=seeds,
                      n_seeds=len(seeds) if seeds else None,
                      raw_seed_ap_mean=float(np.mean([s['test_ap'] for s in seeds])) if seeds else None,
                      raw_seed_ap_std=float(np.std([s['test_ap'] for s in seeds],ddof=1)) if len(seeds)>1 else None)
        for h,head in enumerate(names):
            th = choose_threshold(cy[:,h],cp[:,h])
            far_th = choose_threshold(cy[:,h],cp[:,h],cfg.max_far)
            result['heads'][head] = dict(at_calibration_f1=metrics(ty[:,h],tp[:,h],th),
                                        at_calibration_far=metrics(ty[:,h],tp[:,h],far_th),
                                        raw_ap=ap(ty[:,h],pp['test'][:,h]),
                                        calibration_far_feasible=far_th<=1)
            if h in [0,2,3]:
                result['heads'][head]['events'] = event_report(data,ti,tp[:,h],far_th,3 if h==2 else 1)
            rows.append(dict(model=name,head=head,**result['heads'][head]['at_calibration_f1']))
        # At-risk onset AP removes ongoing floods rather than rewarding easy negatives.
        dry = data['state'][tuple(ti.T)]==0
        result['onset_at_risk_ap']=ap(ty[dry,3],tp[dry,3])
        report[name]=result
        np.savez_compressed(out/f'{name}_predictions.npz',y=ty,p=tp,raw_p=pp['test'],
                            calibration_y=cy,calibration_p=cp,calibration_indices=ci,
                            day=ti[:,0],node=ti[:,1],date=data['dates'][ti[:,0]],node_id=np.array(data['ids'])[ti[:,1]])
    contrasts = {}
    if 'model4' in calibrated:
        for other in ['model2_control','lightgbm','model4_current','discharge_pctl']:
            if other in calibrated:
                contrasts[other] = paired_bootstrap(ty[:,0],calibrated['model4'][:,0],calibrated[other][:,0],
                                                    data['dates'][ti[:,0]],cfg.bootstrap_draws)
    json_save(out/'metrics.json',report); json_save(out/'paired_comparisons.json',contrasts)
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    subgroup = []
    for name,p in calibrated.items():
        for group,values in [('basin',data['basins'][ti[:,1]]),('year',data['dates'][ti[:,0]].astype('datetime64[Y]').astype(str))]:
            for value in np.unique(values):
                sel = values==value
                subgroup.append(dict(model=name,group=group,value=value,n=int(sel.sum()),
                                     flood_ap=ap(ty[sel,0],p[sel,0]),onset_ap=ap(ty[sel,3],p[sel,3])))
    pd.DataFrame(subgroup).to_csv(out/'subgroups.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    for name,p in calibrated.items():
        pr,re,_=precision_recall_curve(ty[:,0],p[:,0])
        axes[0].plot(re,pr,label=f'{name}: AP {ap(ty[:,0],p[:,0]):.3f}')
        order=np.argsort(p[:,0]); bins=np.array_split(order,12)
        axes[1].plot([p[b,0].mean() for b in bins],[ty[b,0].mean() for b in bins],'.-',label=name)
    axes[0].set(xlabel='Recall',ylabel='Precision',title='Held-out 2021–2024: next-day exceedance')
    axes[1].plot([0,1],[0,1],'k--',alpha=.4)
    axes[1].set(xlabel='Mean predicted probability',ylabel='Observed fraction',title='Reliability (equal-count bins)')
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'evaluation.png',dpi=170); plt.close(fig)
    print(pd.DataFrame(rows).query("head == 'flood_1d'")[['model','pr_auc','brier','ece','pod','far']].to_string(index=False))
    return report


def run(root,out,cfg):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    root=Path(root)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    if device.type=='cuda':
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    modules_root=Path(__file__).resolve().parents[1]
    sources=['model4/workflow.py','model2/model.py','model2/modules.py','model2/config.py',
             'floodlib/schema.py','floodlib/blocks.py','floodlib/traincfg.py']
    identity=dict(config=asdict(cfg),data_sha256=file_hash(root/'flood_dataset.parquet'),
                  nodes_sha256=file_hash(root/'nodes.csv'),
                  source_sha256={name:file_hash(modules_root/name) for name in sources})
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature']!=signature:
        raise ValueError('Output contains a different dataset/code/config run. Use a fresh output directory.')
    json_save(manifest,dict(signature=signature,**identity,device=str(device),torch=torch.__version__,
                           numpy=np.__version__,python=platform.python_version(),
                           gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None))
    data=prepare(pd.read_parquet(root/'flood_dataset.parquet'),pd.read_csv(root/'nodes.csv'),cfg)
    json_save(out/'data_audit.json',data['audit']); np.savez_compressed(out/'preprocessing.npz',**data['norm'])
    architectures={}
    for name in ['model4','model2_control','model4_current']:
        model=build_model(name,data,cfg)
        architectures[name]={'parameters':sum(p.numel() for p in model.parameters())}
        if name=='model2_control':
            architectures[name]['config']=asdict(model.net.cfg)
        del model
    json_save(out/'architectures.json',architectures)
    print(json.dumps(data['audit']['split_counts']),flush=True)
    deadline=time.monotonic()+cfg.budget_hours*3600
    preds={}; seed_metrics={}; pending=[]
    # Model 4 first, then the paired architecture control, then a cheap ablation.
    schedule=[('model4',cfg.seeds),('model2_control',cfg.seeds),('model4_current',(cfg.seeds[0],))]
    for name,seeds in schedule:
        collected={k:[] for k in ['calibrate','test']}; details=[]
        for seed in seeds:
            cache=out/f'{name}_seed{seed}_predictions.npz'
            try:
                if cache.exists():
                    with np.load(cache) as saved:
                        pp={k:saved[k] for k in collected}
                else:
                    model,history=fit_seed(name,seed,data,cfg,out,deadline,device)
                    pp={k:predict(model,data,np.argwhere(data['masks'][k]),cfg,device) for k in collected}
                    np.savez_compressed(cache,**pp)
                    del model
                    if device.type=='cuda': torch.cuda.empty_cache()
                for k in collected: collected[k].append(pp[k])
                yy=data['y'][data['masks']['test']]
                details.append(dict(seed=seed,test_ap=ap(yy[:,0],pp['test'][:,0]),onset_ap=ap(yy[:,3],pp['test'][:,3])))
            except BudgetExpired:
                pending.append(f'{name}: seed {seed}')
                print(f'Time budget reached; checkpoint retained for {name} seed {seed}.',flush=True)
        if len(collected['test'])==len(seeds):
            preds[name]={k:np.mean(v,axis=0) for k,v in collected.items()}
            seed_metrics[name]=details
    # Test metrics are saved only after every planned neural run completes.
    # This prevents an incomplete seed prefix being presented as the final run.
    if pending:
        json_save(out/'status.json',dict(complete=False,pending=pending,
                  resume='Rerun with this output directory (or attach extracted outputs as Kaggle input).'))
        print('PARTIAL: no final test comparison. Resume to complete the predeclared experiment.',flush=True)
        return
    print('Neural runs complete. Fitting matched baselines.',flush=True)
    base_cache=out/'baselines.npz'
    if base_cache.exists():
        with np.load(base_cache) as saved:
            for name in ['lightgbm','persistence','discharge_pctl']:
                preds[name]={k:saved[f'{name}_{k}'] for k in ['calibrate','test']}
    else:
        baselines=baseline_predictions(data,cfg,out); preds.update(baselines)
        np.savez_compressed(base_cache,**{f'{name}_{k}':v for name,pp in baselines.items() for k,v in pp.items()})
    evaluate_all(preds,data,cfg,out,seed_metrics)
    json_save(out/'status.json',dict(complete=True,pending=[],
              note='Observed test AP is an experiment result, not a guarantee of superiority or operational readiness.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--out',default='/kaggle/working/model4_runs')
    args=parser.parse_args()
    run(args.root,args.out,Config())
