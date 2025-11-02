#!/usr/bin/env python3
"""
nvml_logger.py - Log NVIDIA GPU power/util/temperature via NVML to CSV.

Usage:
  python nvml_logger.py --interval 0.5 --out gpu_power.csv
"""
import time, csv, argparse, sys
from datetime import datetime
try:
    import pynvml as N
except Exception as e:
    sys.exit("Install pynvml: pip install pynvml")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0, help="sampling interval seconds")
    ap.add_argument("--out", type=str, default="gpu_power.csv", help="output CSV path")
    args = ap.parse_args()

    N.nvmlInit()
    n = N.nvmlDeviceGetCount()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts","gpu_index","power_w","gpu_util_pct","mem_util_pct","temp_c"])
        while True:
            ts = datetime.utcnow().isoformat()
            for i in range(n):
                h = N.nvmlDeviceGetHandleByIndex(i)
                power_mw = N.nvmlDeviceGetPowerUsage(h)  # milliwatts
                util = N.nvmlDeviceGetUtilizationRates(h)
                temp = N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)
                w.writerow([ts, i, round(power_mw/1000,2), util.gpu, util.memory, temp])
            f.flush()
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
