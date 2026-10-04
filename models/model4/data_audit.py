"""Read only required parquet columns and audit sparse image indices, not pixels.

Two builders exist: the chronological image_dataset.csv uses actual pass
dates; the older event-selected image_index.csv must use actual_date, never
the requested target date. Image labels are diagnostics only.
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from floodlib.schema import DYNAMIC_FEATURES


def read_training_panel(path):
    path = Path(path)
    parquet = pq.ParquetFile(path)
    available = parquet.schema_arrow.names
    required = list(dict.fromkeys(['node_id','date','discharge','discharge_pctl',
                                  'valid_sample']+DYNAMIC_FEATURES))
    absent = sorted(set(required)-set(available))
    if absent:
        raise ValueError(f'Missing tabular columns: {absent}')
    optional = ['thr_high','target_flood_1d','target_flood_2d','target_flood_3d','target_onset_1d']
    columns = required+[c for c in optional if c in available]
    frame = pd.read_parquet(path,columns=columns)
    # Forecast tensors have explicit float32 precision; do not materialise a
    # [all samples, lookback, features] array. Windows are sliced per batch.
    for col in columns:
        if col not in ['node_id','date']:
            frame[col] = pd.to_numeric(frame[col],errors='coerce').astype('float32')
    audit = dict(file_bytes=path.stat().st_size,rows=parquet.metadata.num_rows,
                 row_groups=parquet.metadata.num_row_groups,total_columns=len(available),
                 selected_columns=columns,selected_frame_bytes=int(frame.memory_usage(deep=True).sum()))
    return frame,audit


def causal_frame_map(index, node_ids, dates, max_age=14, availability_lag=1):
    """Daily forecast alignment with explicit lag and calendar-day age.

    Default one-day lag is conservative because only dates (not publication
    timestamps) are stored. It is an assumption, not verified availability.
    The returned indices refer to input row positions, even when unsorted.
    """
    if max_age < 0 or availability_lag < 0:
        raise ValueError('Age and availability lag must be nonnegative.')
    date_col = 'actual_date' if 'actual_date' in index else 'date'
    node_col = 'site_id' if 'site_id' in index else 'node_id'
    acq = pd.to_datetime(index[date_col],errors='raise',utc=True).dt.tz_localize(None).dt.normalize()
    if acq.isna().any() or index[node_col].isna().any():
        raise ValueError('Image acquisition dates and site IDs cannot be missing.')
    acquisition = acq.to_numpy('datetime64[D]')
    dates = np.asarray(dates,dtype='datetime64[D]')
    frame = np.full((len(dates),len(node_ids)),-1,np.int32)
    age = np.zeros(frame.shape,np.float32)
    for n,node in enumerate(node_ids):
        positions = np.flatnonzero(index[node_col].astype(str).to_numpy()==str(node))
        positions = positions[np.argsort(acquisition[positions],kind='stable')]
        if not positions.size:
            continue
        available = acquisition[positions]+np.timedelta64(availability_lag,'D')
        latest = np.searchsorted(available,dates,side='right')-1
        has = latest >= 0
        rows = positions[latest.clip(0)]
        days_old = (dates-acquisition[rows]).astype(int)
        has &= days_old <= max_age
        frame[has,n] = rows[has]
        age[has,n] = days_old[has]
    return dict(frame=frame,age=age,present=frame>=0)


def audit_index(path, data):
    index = pd.read_csv(path)
    date_col = 'actual_date' if 'actual_date' in index else 'date'
    node_col = 'site_id' if 'site_id' in index else 'node_id'
    if date_col not in index or node_col not in index:
        return dict(path=str(path),error='No site/acquisition-date columns; index was not used.')
    index = index.copy()
    index['_day'] = pd.to_datetime(index[date_col],errors='coerce',utc=True).dt.tz_localize(None).dt.normalize()
    malformed = index['_day'].isna() | index[node_col].isna()
    invalid_count = int(malformed.sum())
    index = index.loc[~malformed].reset_index(drop=True)
    sites = []
    for node,group in index.groupby(node_col):
        days = np.sort(group['_day'].unique().astype('datetime64[D]'))
        gaps = np.diff(days).astype(int)
        gaps = gaps[gaps>0]
        sites.append(dict(node_id=str(node),frames=len(group),unique_dates=len(days),
                          first_date=str(days[0]),last_date=str(days[-1]),
                          median_gap_days=float(np.median(gaps)) if gaps.size else None,
                          p90_gap_days=float(np.quantile(gaps,.9)) if gaps.size else None,
                          max_gap_days=int(gaps.max()) if gaps.size else None))
    frame_map = causal_frame_map(index,data['ids'],data['dates'])
    coverage = {k:float(frame_map['present'][v].mean()) for k,v in data['masks'].items()}
    # Same-day label consistency is checked against the tabular task, NOT used
    # to choose dates or overwrite either dataset's archived labels.
    join = dict(checked=0,disagreements=0)
    if 'label' in index:
        nmap = {str(n):i for i,n in enumerate(data['ids'])}
        dmap = {str(d):i for i,d in enumerate(data['dates'].astype('datetime64[D]'))}
        for _,row in index.iterrows():
            n=nmap.get(str(row[node_col])); d=dmap.get(str(row['_day'].date()))
            if n is not None and d is not None and pd.notna(row['label']) and np.isfinite(data['state'][d,n]):
                join['checked'] += 1
                join['disagreements'] += int(float(row['label']) != data['state'][d,n])
    numeric_scalars = [c for c in ['water_fraction','vv_mean','vh_mean','valid_fraction'] if c in index]
    return dict(path=str(path),index_bytes=Path(path).stat().st_size,frames=len(index),
                invalid_rows=invalid_count,sites=sites,
                duplicated_site_dates=int(index.duplicated([node_col,'_day']).sum()),
                acquisition_date_column=date_col,
                panel_nodes_with_frames=int(frame_map['present'].any(0).sum()),
                panel_node_count=len(data['ids']),coverage_by_split=coverage,
                frames_by_year=index['_day'].dt.year.value_counts().sort_index().to_dict(),
                label_check=join,available_scalar_columns=numeric_scalars,
                alignment=dict(direction='backward',availability_lag_days=1,max_age_days=14),
                limitations=['One-day availability lag is assumed; publication times are not recorded.',
                             'Image labels are discharge-derived, not independent flood annotations.',
                             'Pixel files were not loaded or used by this tabular experiment.'])


def audit_images(data, roots=None):
    if roots is None:
        roots = [Path('/kaggle/input')]
        if os.environ.get('SAR_DATA_ROOT'):
            roots.append(Path(os.environ['SAR_DATA_ROOT']))
    paths = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            paths.add(root)
        elif root.exists():
            for name in ['image_dataset.csv','image_index.csv']:
                paths.update(root.rglob(name))
    reports = []
    for path in sorted(paths):
        try:
            reports.append(audit_index(path,data))
        except (ValueError,KeyError,pd.errors.ParserError) as error:
            reports.append(dict(path=str(path),error=str(error)))
    return dict(indices=reports,found=bool(paths),
                note='Metadata audit only. Attach uom230429e/flood-data-set to audit actual acquisition gaps.')
