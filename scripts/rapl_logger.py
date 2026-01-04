#!/usr/bin/env python3
"""
rapl_logger.py - Log CPU package/DRAM power via Intel RAPL to CSV (Linux).

Usage:
  sudo python rapl_logger.py --interval 0.5 --out cpu_power.csv
"""
import time, csv, argparse, glob, os, sys
from datetime import datetime

def read_uj(path):
    with open(path, "r") as f:
        return int(f.read().strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="cpu_power.csv")
    args = ap.parse_args()

    domains = sorted(glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"))
    if not domains:
        sys.exit("No RAPL domains found. Run on Intel platform with RAPL enabled.")
    last = [read_uj(p) for p in domains]; last_t = time.time()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        header = ["ts"] + [f"domain_{i}_w" for i in range(len(domains))]
        w.writerow(header)
        while True:
            time.sleep(args.interval)
            now = time.time()
            now_vals = [read_uj(p) for p in domains]
            watts = [(now_vals[i]-last[i]) / (now-last_t) / 1e6 for i in range(len(domains))]
            w.writerow([datetime.utcnow().isoformat()] + [round(x,2) for x in watts])
            f.flush()
            last, last_t = now_vals, now

if __name__ == "__main__":
    main()
