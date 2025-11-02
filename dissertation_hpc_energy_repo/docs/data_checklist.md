# Data Checklist (Gold / Silver / Bronze)

Use this to scope what can be shared. All identifiers must be **hashed**; paths/job names redacted.

## A. Time Window & Granularity
- Period: **90–180 days** (rolling is fine).  
- Clock: **UTC** preferred (or specify local tz).  
- Granularity: job events (event‑time); telemetry **1–5 s** (aggregated okay).

## B. Workload & Scheduler (Job‑level)
**Format:** SWF / CSV / Parquet  
**Fields (de‑identified):**
- `job_id_hash`, `submit_ts`, `start_ts`, `end_ts`, `exit_status`  
- requested vs used: `req_nodes`, `req_cores`, `req_gpus`, `req_mem_gb`, `req_time_s`; `cpu_core_seconds`, `gpu_seconds`, `mem_gb_seconds`  
- context: `qos/partition`, `preempted_flag`, `priority`, `project_id_hash`, `user_id_hash`  
**Nice‑to‑have:** container hash, framework hint (MPI/Torch), reason codes

## C. Node/Fabric Inventory (Static)
- `node_id_hash`, `cpu_model`, `gpu_model`, `gpu_count`, `mem_gb`, `hbm_gb`  
- `rack_zone` (coarse), `interconnect`, `node_power_cap_w`

## D. Telemetry & Power (Time‑Series)
- CPU power (RAPL), GPU power (NVML), cpu/gpu util%, mem bandwidth%, temperature  
- Power cap / DVFS state  
- Join by `ts + node_id_hash` (or window‑aligned if direct join not permitted).

## E. Scheduler Events
- Preemptions, migrations, backfills, power‑cap changes, maintenance windows.

## F. Carbon/Energy Price (Optional)
- Grid carbon intensity time series (region); energy tariff schedule.

## G. Anonymisation & Access
- Hash user/project/node/image IDs (consistent).  
- Redact sensitive paths/job names/locations.  
- Delivery via S3/GCS/Azure Blob/SFTP or enclave/on‑prem processing.

## H. Sharing Tiers
- **Gold:** Jobs + per‑node telemetry (1–5 s) + inventory + events.  
- **Silver:** Jobs + aggregated telemetry (per‑minute or per‑rack) + coarse inventory.  
- **Bronze:** Jobs only (SWF); energy inferred via surrogates.

---

**What you receive in return:**  
Private report (utilisation, energy, **CO₂e**, hotspots), baseline vs. AI policy, and reusable tools/configs.
