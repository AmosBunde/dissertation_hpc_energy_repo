#!/usr/bin/env python3
"""
carbon_intensity_gb.py - Fetch GB carbon-intensity time series.

Usage:
  python carbon_intensity_gb.py --from 2024-01-01 --to 2024-01-07 --out carbon.csv
"""
import argparse, requests, csv

API = "https://api.carbonintensity.org.uk/intensity"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--to", dest="end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", default="carbon.csv")
    args = ap.parse_args()

    url = f"{API}/{args.start}/{args.end}"
    import urllib.request, json
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode("utf-8"))["data"]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["from","to","intensity_forecast","intensity_actual"])
        for d in data:
            w.writerow([d["from"], d["to"], d["intensity"]["forecast"], d["intensity"].get("actual")])

if __name__ == "__main__":
    main()
