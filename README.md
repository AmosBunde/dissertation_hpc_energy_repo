# AI-Driven Workload & Energy Optimization for Exascale Scientific Computing

_Dissertation repository for Amos Ochieng’ Bunde (MSc → PhD track)._  
Theme: **Energy-aware scheduling** and **carbon-aware operations** for heterogeneous CPU/GPU clusters.  
Core idea: combine **surrogate predictors** (runtime, energy) with a **reinforcement learning (RL)** policy that optimizes energy/CO₂e subject to throughput/SLA constraints.

[![Build](https://img.shields.io/badge/CI-passing-brightgreen)](#)

---

## Quickstart (10-minute smoke test)

```bash
# 0) Create a virtual env and install deps
make setup

# 1) Get a REAL PWA trace (examples below) and convert to CSV
mkdir -p traces data/processed
curl -L -o traces/LLNL-Atlas-2006-2.1-cln.swf.gz   https://www.cs.huji.ac.il/labs/parallel/workload/l_llnl_atlas/LLNL-Atlas-2006-2.1-cln.swf.gz
gunzip -f traces/LLNL-Atlas-2006-2.1-cln.swf.gz

# Convert SWF → CSV jobs table
python scripts/swf_to_csv.py --swf traces/LLNL-Atlas-2006-2.1-cln.swf --out data/processed/llnl_jobs.csv

# 2) Build features (clean + leakage-safe) and train both surrogates
make train
# (equivalent to: make features && make train-runtime && make train-energy)

# 3) Train a tiny PPO policy on the minimal RL env (proof of wiring)
make rl-train       # uses Gymnasium + Stable-Baselines3

# 4) Render the plotting notebook to HTML (queue backlog, energy, CO₂e)
make plot           # -> notebooks/plot_report.html
```

---

## Data flow (at a glance)

```
SWF trace → swf_to_csv.py → data/processed/*_jobs.csv
         → build_features.py (clean + leakage-safe features) → data/processed/*_features.csv
         → train_*_surrogate.py → models/*.joblib (used by RL env/simulator)
```

---

## Repository layout

```
.
├─ README.md
├─ Makefile                          # setup, simulate, plot, features, train, rl-train
├─ requirements.txt / environment.yml
├─ .github/workflows/ci.yml
├─ docs/                             # overview, data request, checklist, ethics
├─ notebooks/
│  └─ plot_backlog_energy_co2e.ipynb
├─ scripts/
│  ├─ swf_to_csv.py                  # SWF → CSV (jobs)
│  ├─ build_features.py              # clean + leakage-safe features → runtime/energy feature CSVs
│  ├─ train_energy_surrogate.py      # trains energy surrogate (RF)
│  ├─ train_runtime_surrogate.py     # trains runtime surrogate (RF)
│  ├─ nvml_logger.py                 # GPU power (NVML) → CSV (optional)
│  ├─ rapl_logger.py                 # CPU power (RAPL) → CSV (optional)
│  └─ carbon_intensity_gb.py         # UK Grid carbon intensity → CSV (optional)
├─ rl/
│  ├─ env.py                         # minimal Gymnasium env using surrogates
│  └─ train_ppo.py                   # train PPO (Stable-Baselines3)
├─ batsim/
│  ├─ platform.xml
│  ├─ config_llnl.json
│  ├─ config_kth.json
│  └─ run_replay.sh                  # placeholder for real batsim invocation
├─ traces/
│  ├─ pwa_llnl.swf  (placeholder — replace with real PWA SWF)
│  └─ pwa_kth.swf   (placeholder — replace with real PWA SWF)
├─ data/
│  ├─ raw/                           # telemetry, carbon series
│  └─ processed/                     # *_jobs.csv, *_features.csv
└─ models/                           # *.joblib (surrogates), PPO policies
```

---

## Set up the environment

```bash
make setup
# or, with conda:
# conda env create -f environment.yml && conda activate hpc-energy
```

Key packages: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `jupyter`, `stable-baselines3`, `gymnasium`, `pynvml` (optional for NVML logging).

> If you see `ModuleNotFoundError: gym`, we use **Gymnasium**. `make setup` installs it, but you can also run:  
> `pip install gymnasium stable-baselines3`

---

## Getting real traces (download → convert)

**Note:** `traces/pwa_llnl.swf` and `traces/pwa_kth.swf` in this repo are placeholders. Replace them with real logs from the **Parallel Workloads Archive (PWA)**.

### Examples (copy/paste)

```bash
mkdir -p traces data/processed

# LLNL Atlas (cleaned)
curl -L -o traces/LLNL-Atlas-2006-2.1-cln.swf.gz   https://www.cs.huji.ac.il/labs/parallel/workload/l_llnl_atlas/LLNL-Atlas-2006-2.1-cln.swf.gz
gunzip -f traces/LLNL-Atlas-2006-2.1-cln.swf.gz
python scripts/swf_to_csv.py --swf traces/LLNL-Atlas-2006-2.1-cln.swf --out data/processed/llnl_jobs.csv

# SDSC SP2 (cleaned)
curl -L -o traces/SDSC-SP2-1998-4.2-cln.swf.gz   https://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_sp2/SDSC-SP2-1998-4.2-cln.swf.gz
gunzip -f traces/SDSC-SP2-1998-4.2-cln.swf.gz
python scripts/swf_to_csv.py --swf traces/SDSC-SP2-1998-4.2-cln.swf --out data/processed/llnl_jobs.csv

# KTH SP2 (cleaned)
curl -L -o traces/KTH-SP2-1996-2.1-cln.swf.gz   https://www.cs.huji.ac.il/labs/parallel/workload/l_kth_sp2/KTH-SP2-1996-2.1-cln.swf.gz
gunzip -f traces/KTH-SP2-1996-2.1-cln.swf.gz
python scripts/swf_to_csv.py --swf traces/KTH-SP2-1996-2.1-cln.swf --out data/processed/kth_jobs.csv
```

**Sanity check:**
```bash
grep -v '^;' traces/LLNL-Atlas-2006-2.1-cln.swf | sed '/^[[:space:]]*$/d' | head      # preview jobs
tail -n +2 data/processed/llnl_jobs.csv | wc -l                                      # count rows
```

---

## Step-by-step: From traces to surrogates

### 1) SWF → jobs table
```bash
python scripts/swf_to_csv.py --swf traces/LLNL-Atlas-2006-2.1-cln.swf --out data/processed/llnl_jobs.csv
```
- Input: real `.swf` in `traces/`
- Output: `data/processed/llnl_jobs.csv` (submit, wait, run, req_procs, …)

### 2) Clean + leakage-safe features
```bash
python scripts/build_features.py --jobs_csv data/processed/llnl_jobs.csv --synthetic-energy
```
Generates:
- `data/processed/runtime_features.csv` (features + `runtime_s`)
- `data/processed/energy_features.csv` (features + `energy_j`, synthetic unless you provide labels)

**Leakage-safe** = only what’s knowable at scheduling time (`req_procs`, `age_in_queue_s`, `submit_hour`, `submit_dow`).

### 3) Train the surrogates (runtime & energy)

```bash
# Runtime surrogate (target = runtime_s)
python scripts/train_runtime_surrogate.py --csv data/processed/runtime_features.csv --target runtime_s --out models/runtime_surrogate_rf.joblib

# Energy surrogate (target = energy_j)
python scripts/train_energy_surrogate.py  --csv data/processed/energy_features.csv  --target energy_j   --out models/energy_surrogate_rf.joblib
```

**Shortcut:** do all steps with:
```bash
make train
```

---

## Optional: Collect power & carbon data (to replace synthetic labels later)

GPU power via NVML:
```bash
python scripts/nvml_logger.py --interval 0.5 --out data/raw/gpu_power.csv
```

CPU power via RAPL (Linux):
```bash
sudo python scripts/rapl_logger.py --interval 0.5 --out data/raw/cpu_power.csv
```

Carbon intensity (GB example):
```bash
python scripts/carbon_intensity_gb.py --from 2025-01-01 --to 2025-01-07 --out data/raw/carbon_gb.csv
```

Then align power samples to job windows to produce real `energy_j` labels and retrain the energy surrogate.

---

## RL: Training a policy with the surrogates

`rl/env.py` is a **minimal Gymnasium environment** that:
- Loads `models/runtime_surrogate_rf.joblib` & `models/energy_surrogate_rf.joblib`
- Observes top-K jobs’ features + backlog
- Action: pick which job to start next (simple)
- Reward: penalizes **kWh** and backlog

Train PPO (Stable-Baselines3):
```bash
make rl-train      # or: make rl-train-fast
```

Output policy: `models/ppo_hpcenv.zip`.

---

## Simulation and plotting

Replay with your preferred simulator (placeholder Batsim runner):
```bash
make simulate-LLNL
make simulate-KTH
```

Render the notebook with plots (queue backlog, energy, CO₂e):
```bash
make plot
# Produces notebooks/plot_report.html
```

---

## Makefile targets (reference)

```makefile
setup             # venv + pip install
simulate          # run batsim/run_replay.sh with $(CONFIG)
simulate-LLNL     # uses batsim/config_llnl.json
simulate-KTH      # uses batsim/config_kth.json
plot              # executes notebook and writes plot_report.html

features          # clean + leakage-safe features from *_jobs.csv
train-runtime     # train runtime surrogate from runtime_features.csv
train-energy      # train energy surrogate from energy_features.csv
train             # features + train-runtime + train-energy

rl-train          # train PPO on minimal env (proof of wiring)
rl-train-fast     # shorter training run
clean             # tidy artifacts
```

---

## Troubleshooting

- **CSV empty (only header):** The SWF was a placeholder or only comments. Use a real PWA file and reconvert.  
- **`No rule to make target 'setup'`:** Run from repo root where `Makefile` lives.  
- **`ModuleNotFoundError: gym`**: We use **Gymnasium**. Ensure `pip install gymnasium stable-baselines3` or run `make setup` again.  
- **Notebook not found**: Ensure `notebooks/plot_backlog_energy_co2e.ipynb` exists (it’s included).  
- **Slow/timeout on plot**: Increase `ExecutePreprocessor.timeout` in the `Makefile` `plot` target.

---

## Appendix: Terminology & Concepts

**SWF (Standard Workload Format)** — HPC job logs; one job per line with submit, wait, run, allocated/requested processors.  
**Surrogate Predictors** — fast supervised models approximating runtime/energy so the scheduler can do “what-if” planning.  
**RL (Reinforcement Learning)** — learn scheduling decisions from reward signals over time.  
**PPO (Proximal Policy Optimization)** — robust policy-gradient algorithm used here.  
**Gymnasium** — RL environment API used by Stable-Baselines3.  
**NVML (NVIDIA Management Library)** — GPU telemetry (incl. power) for real energy labels.  
**RAPL (Running Average Power Limit)** — CPU energy counters/limits on Intel platforms.  
**DVFS** — adjust device frequency/voltage to save power.  
**Power Cap** — set a device/node power ceiling to meet energy/carbon budgets.  
**kWh / Joule** — energy units; 1 kWh = 3.6e6 J.  
**CO₂e** — emissions; computed from energy × grid carbon intensity.  
**Backfilling** — fill idle gaps with smaller jobs while preserving reservations.  
**SLA/SLO** — user-facing time/throughput objectives; the policy respects these.  
**MAPE/MAE** — model error metrics; used to evaluate surrogates.  
**Explainability (e.g., SHAP)** — understand predictions or policy decisions for operator trust.  
**k-Anonymity** — privacy; ensure aggregated/blurred telemetry where required.

---

## Ethics & Data Handling (summary)

- IDs (user/project/node) are **hashed**.  
- Telemetry is **aggregated** (1–60s) when shared.  
- Results sent back to providers; public artifacts reviewed for privacy.  
- See `docs/ethics_and_privility.md` for details.
