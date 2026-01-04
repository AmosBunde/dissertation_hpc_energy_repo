# Dissertation Overview
**Title:** AI‑Driven Workload & Energy Optimization for Exascale Scientific Computing  
**Candidate:** Amos Ochieng’ Bunde • MSc in Information Systems & Computing • Supervisor: Mahmoud Elbattah  
**Date:** 2025-11-02

---

## 1. Background & Motivation
Exascale systems enable breakthroughs in climate modeling, life sciences, and AI training, but they come with sharp **energy and carbon footprints**. Conventional schedulers (e.g., FCFS, EASY backfilling) optimise for throughput and fairness, not **energy** or **CO₂e**. At the same time, modern workloads (GPU‑heavy DL, mixed MPI jobs) require **heterogeneous, bursty** resources that stress traditional policies.

## 2. Problem Statement
How can we **predict** job runtime and energy consumption and use those predictions inside a **policy** that optimises **performance × energy × carbon**—without breaching SLAs/latency?

## 3. Research Objectives
1. **Modeling:** Train **runtime** and **energy** predictors from workload traces + telemetry (CPU/GPU power).  
2. **Scheduling:** Design a **hybrid optimisation** approach (Constrained RL + Bayesian Optimisation) for policy selection and power‑cap/DVFS control.  
3. **Evaluation:** Replay real traces (PWA, Google/Alibaba slices) in a simulator and compare to FCFS/EASY/energy heuristics.  
4. **Explainability:** Provide operator‑interpretable rationales (feature attribution, counterfactuals) for policy decisions.  
5. **Carbon‑awareness:** Convert kWh to **CO₂e** with regional intensity time series; study carbon‑aware deferral/placement.

## 4. Research Questions (RQs)
- **RQ1:** Can AI‑assisted scheduling reduce **energy (kWh)** and **CO₂e** without degrading throughput?  
- **RQ2:** How well do runtime/energy predictors **generalise** across workloads and architectures?  
- **RQ3:** What are the trade‑offs between **queue latency**, **throughput**, and **energy** under different caps/policies?  
- **RQ4:** What **explanations** make operator adoption likely in production HPC?

## 5. Methodology (Summary)
- **Data:** PWA SWF traces; Google 2019 BigQuery slice; optional onsite NVML/RAPL logs for calibration.  
- **Models:** Gradient boosting & GNNs for runtime; surrogate energy model from NVML/RAPL.  
- **Policy:** Constrained RL for online actions; BO for policy knobs (caps, batching, backfill windows).  
- **Simulator:** Batsim/SimGrid for trace replay; energy model integrated.  
- **Metrics:** kWh, **CO₂e**, p95/p99 wait, throughput, preemption count; operator explanation quality.

## 6. Contributions
- **C1:** Public **tooling stack** for energy/carbon‑aware scheduling research.  
- **C2:** Empirical results across multiple public traces.  
- **C3:** Operator‑facing **explanations** and **playbooks** for adoption.  
- **C4:** Reproducible configs, datasets (where permitted), and notebooks.

## 7. Risks & Mitigations
- **No power telemetry:** use proxy apps + NVML/RAPL to train surrogates; validate with sensitivity analysis.  
- **Trace mismatch:** run multiple traces + ablations.  
- **Policy instability:** bootstrap from imitation learning and add constraints.

## 8. Timeline (high level)
- **Months 1–2:** Literature, data ingestion, baseline simulators.  
- **Months 3–6:** Predictors (runtime/energy), evaluation baselines.  
- **Months 7–10:** RL/BO policy + explainability.  
- **Months 11–12:** Write‑up; artifacts.

---

## 9. Repo Pointers
- `batsim/` configs; `traces/` PWA placeholders; `notebooks/plot_backlog_energy_co2e.ipynb`; `scripts/*` for telemetry/carbon.
