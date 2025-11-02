
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

# 1) Convert an SWF trace (PWA) to CSV jobs table
python scripts/swf_to_csv.py --swf traces/pwa_llnl.swf --out data/processed/llnl_jobs.csv

# 2) Build features (clean + leakage-safe) and train both surrogates
make train
# (equivalent to: make features && make train-runtime && make train-energy)

# 3) Train a tiny PPO policy on the minimal RL env (proof of wiring)
make rl-train

# 4) Render the plotting notebook to HTML (queue backlog, energy, CO₂e)
make plot
# opens notebooks/plot_report.html on your machine (depending on OS you’ll open it manually)
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
│  ├─ train_runtime_surrogate.py     # copy of energy trainer, target=runtime_s
│  ├─ nvml_logger.py                 # GPU power (NVML) → CSV (optional)
│  ├─ rapl_logger.py                 # CPU power (RAPL) → CSV (optional)
│  └─ carbon_intensity_gb.py         # UK Grid carbon intensity → CSV (optional)
├─ rl/
│  ├─ env.py                         # minimal Gym env that uses the surrogates
│  └─ train_ppo.py                   # train PPO on env.py (Stable-Baselines3)
├─ batsim/
│  ├─ platform.xml
│  ├─ config_llnl.json
│  ├─ config_kth.json
│  └─ run_replay.sh                  # placeholder for real batsim invocation
├─ traces/
│  ├─ pwa_llnl.swf  (placeholder – drop real PWA file here)
│  └─ pwa_kth.swf   (placeholder – drop real PWA file here)
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

---

## Step-by-step: From traces to surrogates

### 1) SWF → jobs table
```bash
python scripts/swf_to_csv.py --swf traces/pwa_llnl.swf --out data/processed/llnl_jobs.csv
```

- Input: `traces/pwa_llnl.swf` (Standard Workload Format from PWA)
- Output: `data/processed/llnl_jobs.csv` (submit, wait, run, req_procs, …)

### 2) Clean + leakage-safe features
```bash
python scripts/build_features.py --jobs_csv data/processed/llnl_jobs.csv --synthetic-energy
```
Generates:
- `data/processed/runtime_features.csv` (features + `runtime_s`)
- `data/processed/energy_features.csv` (features + `energy_j`, synthetic unless you provide labels)

**Leakage-safe** means: inputs are only what’s knowable at scheduling time (e.g., `req_procs`, `age_in_queue_s`, `submit_hour`, `submit_dow`). No post-hoc usage/outcomes in features.

### 3) Train the surrogates (runtime & energy)

Create a runtime trainer by copying the energy trainer:
```bash
cp scripts/train_energy_surrogate.py scripts/train_runtime_surrogate.py
```

Now train:

```bash
# Runtime surrogate (target = runtime_s)
python scripts/train_runtime_surrogate.py --csv data/processed/runtime_features.csv --target runtime_s
# → models/runtime_surrogate_rf.joblib

# Energy surrogate (target = energy_j)
mkdir -p models
python scripts/train_energy_surrogate.py --csv data/processed/energy_features.csv --target energy_j
# → models/energy_surrogate_rf.joblib
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

> Later, you’ll time-align power samples to job start/end to form true `energy_j` labels and retrain the energy surrogate.

---

## RL: Training a policy with the surrogates

`rl/env.py` is a **minimal Gym environment** that:
- Loads `models/runtime_surrogate_rf.joblib` & `models/energy_surrogate_rf.joblib`
- Observes top-K jobs’ features + backlog
- Action = pick which job to start next (simple)
- Reward penalizes **kWh** and backlog

Train PPO (Stable-Baselines3):
```bash
# Ensure surrogates exist (make train), then:
make rl-train
# or faster:
make rl-train-fast
```

This produces a policy (e.g., `models/ppo_hpcenv.zip`).  
It’s intentionally small so you can **verify end-to-end** quickly; you’ll later expand actions (power caps/DVFS/placement) and transition dynamics (backfilling, contention, real simulator I/O).

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
# Produces notebooks/plot_report.html (executed from plot_backlog_energy_co2e.ipynb)
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

- **No `llnl_jobs.csv`**: Run the SWF conversion step first.  
- **Energy labels missing**: Use `--synthetic-energy` to bootstrap; replace later with real NVML/RAPL integration.  
- **RL install errors**: Ensure `stable-baselines3` and `gymnasium` installed (`make setup`).  
- **Notebook render takes long**: It’s executed headless; adjust `ExecutePreprocessor.timeout` in the Makefile if needed.

---

## Appendix: Terminology & Concepts

**SWF (Standard Workload Format)**  
Plain-text format for HPC job logs; each line is one job with fields like submit time, wait time, run time, and allocated/requested processors.

**Surrogate Predictors**  
Fast supervised models that approximate hard-to-measure or expensive outcomes (e.g., runtime, energy). Trained once from historical data (or telemetry) and queried many times during policy search or what-if analysis.

**RL (Reinforcement Learning)**  
A learning paradigm for sequential decision-making: an agent observes state, takes actions, and receives rewards. Here, actions include selecting jobs and (later) power caps/placement; rewards penalize energy/CO₂e and SLA violations.

**PPO (Proximal Policy Optimization)**  
A stable, widely used policy-gradient RL algorithm. Good default for discrete/continuous control—used here to train the scheduler policy.

**Gymnasium (Gym)**  
Common interface for RL environments (obs/action spaces, `step`, `reset`). Lets you plug your problem into many RL libraries quickly.

**NVML (NVIDIA Management Library)**  
API to read GPU telemetry (power, utilization, temperature). We use it to collect power data for energy labels.

**RAPL (Running Average Power Limit)**  
Intel interface to read energy counters (package/DRAM) and apply power limits on CPUs. Used to estimate CPU power/energy during job runs.

**DVFS (Dynamic Voltage and Frequency Scaling)**  
Technique to adjust frequency/voltage (and indirectly power) of CPU/GPU devices; key control knob for energy savings in schedulers.

**Power Cap**  
Upper bound on device or node power; schedulers can set this to meet energy or thermal budgets (or to shift compute to low-carbon windows).

**kWh / Joule**  
Energy units. 1 kWh = 3.6e6 Joules. Our surrogate predicts **J**; we convert to **kWh**.

**CO₂e (Carbon Dioxide Equivalent)**  
Emissions metric. We compute: `kWh × carbon_intensity (gCO2/kWh)` → grams, then convert to kg or tons.

**Backfilling**  
Scheduling technique that fills idle slots with smaller jobs while preserving reservations for large jobs; improves utilization/throughput.

**SLA / SLO (Service-Level Agreement / Objective)**  
User-facing performance targets (e.g., p95 wait time). Our reward can penalize violations.

**MAPE / MAE (Error metrics)**  
Model evaluation metrics for surrogate predictors (runtime/energy). MAPE: mean absolute percentage error; MAE: mean absolute error.

**Explainability (e.g., SHAP)**  
Tools that attribute predictions (or policy choices) to features; helps operators trust and tune the AI-assisted scheduler.

**k-Anonymity**  
Privacy technique to ensure any data point is indistinguishable from at least k-1 others. We report aggregates and review with data providers.

---

## Ethics & Data Handling (summary)

- All IDs (user/project/node) consistently **hashed**.  
- Telemetry shared at **aggregated** cadence when needed (1–60s).  
- Confidential results sent back to providers; public outputs pre-reviewed.  
- See `docs/ethics_and_privility.md` for full policy.
