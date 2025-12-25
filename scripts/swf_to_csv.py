#!/usr/bin/env python3
"""
swf_to_csv.py — Convert a Standard Workload Format (SWF) file to CSV.

Usage:
  python scripts/swf_to_csv.py --swf traces/pwa_llnl.swf --out data/processed/llnl_jobs.csv
"""
import argparse, csv, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swf", required=True, help="path to .swf workload file")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cols = [
        "job","submit","wait","run","alloc_procs","avg_cpu","used_mem",
        "req_procs","req_time","req_mem","status","uid","gid","appid",
        "queue","partition","preceding_job","think_time"
    ]

    with open(args.swf, "r") as fin, open(args.out, "w", newline="") as fout:
        w = csv.writer(fout); w.writerow(cols)
        for line in fin:
            s = line.strip()
            if not s or s.startswith(";"):
                continue
            parts = s.split()
            if len(parts) < len(cols):
                parts += ["-1"] * (len(cols) - len(parts))
            w.writerow(parts[:len(cols)])

if __name__ == "__main__":
    main()
