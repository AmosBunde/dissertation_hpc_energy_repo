
#!/usr/bin/env python3
import argparse, os, sys
import pandas as pd
import numpy as np

def coerce_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs_csv', default='data/processed/llnl_jobs.csv')
    ap.add_argument('--outdir', default='data/processed')
    ap.add_argument('--synthetic-energy', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.jobs_csv):
        print(f"[ERROR] jobs_csv not found: {args.jobs_csv}", file=sys.stderr)
        print("Generate it first, e.g.:", file=sys.stderr)
        print("  python scripts/swf_to_csv.py --swf traces/pwa_llnl.swf --out data/processed/llnl_jobs.csv", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.jobs_csv)

    # Replace -1 with NaN and coerce numerics
    df = df.replace(-1, np.nan)
    num_cols = ['submit','wait','run','alloc_procs','req_procs','req_time','req_mem']
    df = coerce_num(df, num_cols)

    # Sanity checks
    df['wait'] = df['wait'].clip(lower=0) if 'wait' in df.columns else np.nan
    if 'run' in df.columns:
        df = df[df['run'].fillna(0) > 0]

    # Targets
    df['runtime_s'] = df['run']

    # Leakage-safe inputs
    if 'req_procs' not in df.columns or df['req_procs'].isna().all():
        df['req_procs'] = df.get('alloc_procs', pd.Series(0, index=df.index))

    if 'submit' in df.columns:
        df['submit'] = df['submit'].fillna(0)
        df['submit_hour'] = ((df['submit']/3600) % 24).astype(int)
        df['submit_dow'] = ((df['submit']/86400) % 7).astype(int)
    else:
        df['submit_hour'] = 0
        df['submit_dow'] = 0

    df['age_in_queue_s'] = df['wait'].fillna(0)

    runtime_features = pd.DataFrame({
        'req_procs': df['req_procs'].fillna(0),
        'age_in_queue_s': df['age_in_queue_s'],
        'submit_hour': df['submit_hour'],
        'submit_dow': df['submit_dow'],
        'runtime_s': df['runtime_s']
    }).dropna(subset=['runtime_s'])

    energy_df = pd.DataFrame({
        'req_procs': df['req_procs'].fillna(0),
        'age_in_queue_s': df['age_in_queue_s'],
        'submit_hour': df['submit_hour'],
        'submit_dow': df['submit_dow'],
    })

    if 'energy_j' in df.columns and df['energy_j'].notna().any():
        energy_df['energy_j'] = df['energy_j']
    else:
        if not args.synthetic_energy:
            print("[ERROR] No energy_j labels. Re-run with --synthetic-energy or provide labels.", file=sys.stderr)
            sys.exit(2)
        c0, c1 = 20.0, 0.5
        energy_df['energy_j'] = df['runtime_s'] * (c0 + c1*df['req_procs'].fillna(0))

    # Outlier clipping
    runtime_q_hi = runtime_features['runtime_s'].quantile(0.999)
    runtime_features = runtime_features[runtime_features['runtime_s'] <= runtime_q_hi]
    energy_q_hi = energy_df['energy_j'].quantile(0.999)
    energy_df = energy_df[energy_df['energy_j'] <= energy_q_hi]

    out_rt = os.path.join(args.outdir, 'runtime_features.csv')
    out_en = os.path.join(args.outdir, 'energy_features.csv')
    runtime_features.to_csv(out_rt, index=False)
    energy_df.to_csv(out_en, index=False)
    print(f"[OK] {out_rt} -> {len(runtime_features)} rows")
    print(f"[OK] {out_en} -> {len(energy_df)} rows")

if __name__ == '__main__':
    main()
