# AI‑Driven Workload & Energy Optimization for Exascale Scientific Computing

> Dissertation repository for Amos Ochieng’ Bunde (MSc → PhD track).  
> Theme: **Energy‑aware, AI‑assisted scheduling** and **carbon‑aware operations** for HPC and AI training/inference at (pre‑)exascale.

[![CI](https://github.com/USER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/USER/REPO/actions/workflows/ci.yml)

---

## 📌 Problem
Exascale systems deliver unprecedented performance but at a steep **energy and carbon cost**. Traditional schedulers (FCFS/EASY) are throughput‑oriented, rarely **energy‑aware** or **carbon‑aware**, and struggle with heterogeneous CPU/GPU clusters used for modern AI workloads.

## 🎯 Objectives
1. **Predict** job runtime and energy using workload traces + telemetry.
2. **Optimize** scheduling via **RL/BO hybrid** policies under energy/carbon constraints.
3. **Evaluate** on real traces (PWA/Google/Alibaba) via **trace replay**.
4. **Explain** operator decisions and quantify **kWh/CO₂e** trade‑offs.

## 🧱 Repository Structure
```
.
├─ README.md
├─ Makefile
├─ requirements.txt
├─ environment.yml
├─ .gitignore
├─ LICENSE
├─ CITATION.cff
├─ CONTRIBUTING.md
├─ CODE_OF_CONDUCT.md
├─ docs/
│  ├─ dissertation_overview.md
│  ├─ data_request_letter.md
│  ├─ data_checklist.md
│  └─ ethics_and_privacy.md
├─ notebooks/
│  └─ plot_backlog_energy_co2e.ipynb
├─ scripts/
│  ├─ nvml_logger.py
│  ├─ rapl_logger.py
│  ├─ carbon_intensity_gb.py
│  ├─ train_energy_surrogate.py
│  └─ export_google_trace.sql
├─ batsim/
│  ├─ platform.xml
│  ├─ config_llnl.json
│  ├─ config_kth.json
│  └─ run_replay.sh
├─ traces/
│  ├─ pwa_llnl.swf  (placeholder – drop real PWA file here)
│  └─ pwa_kth.swf   (placeholder – drop real PWA file here)
├─ data/
│  ├─ raw/
│  └─ processed/
└─ experiments/
   └─ README.md
```

## 🚀 Quickstart
```bash
# 1) Clone or unzip the repo locally
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Drop real PWA SWF files
# traces/pwa_llnl.swf, traces/pwa_kth.swf

# 3) Run baseline simulations (placeholder runner; wire to your batsim command)
make simulate-LLNL
make simulate-KTH

# 4) Render plots (queue backlog, energy, CO₂e)
TRACE_PATH=traces/pwa_llnl.swf make plot
open notebooks/plot_report.html  # or xdg-open on Linux
```

## 🧪 Reproducibility
- All analysis notebooks limited to **deterministic seeds** where applicable.
- CI checks for style, notebook execution, and basic linting (see `.github/workflows/ci.yml`).

## 🔒 Data & Ethics
- No PII; user/project/node identifiers are **hashed**.  
- Telemetry may be **aggregated** (1–60 s).  
- Results reported with **k‑anonymity** and reviewed by providers pre‑publication.  
- See `docs/ethics_and_privacy.md` and `docs/data_checklist.md`.

## 🔗 Public Datasets (for immediate use)
- **PWA SWF traces**: https://www.cs.huji.ac.il/labs/parallel/workload/  
- **Google Cluster 2019**: BigQuery public dataset (export with `scripts/export_google_trace.sql`)  
- **Carbon intensity (GB)**: https://api.carbonintensity.org.uk/

## 📣 Citation
If you use this repo, please cite (see `CITATION.cff`).

## 📜 License
MIT (see `LICENSE`).

---

### GitHub Setup
```bash
git init
git add .
git commit -m "Initial commit: dissertation repo skeleton"
git branch -M main
git remote add origin git@github.com:<your-user>/<your-repo>.git
git push -u origin main
```
