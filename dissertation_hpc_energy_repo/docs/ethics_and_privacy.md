# Ethics & Privacy

## Principles
- **No PII.** All `user_id`, `project_id`, and `node_id` values are **hashed** with consistent salts.
- **Least privilege.** Data stored in an encrypted, access‑controlled bucket; access restricted to named researchers.
- **Aggregation by default.** Where raw telemetry is sensitive, use **1–60 s** aggregates.
- **No re‑identification.** We will not attempt to re‑identify users or projects.
- **k‑Anonymity in reporting.** Public plots/tables redact sparse categories and show aggregates only.

## Data Handling
- **Ingress:** via signed URLs/SFTP; checksum verification on arrival.  
- **Storage:** encrypted at rest; network isolation; audit logging.  
- **Processing:** reproducible pipelines; versioned configs and notebooks.  
- **Egress:** private reports shared back to providers; any public artifact is pre‑reviewed by the provider.

## Retention & Deletion
- Retention default **≤ 12 months** (or per provider policy).  
- Certified deletion on request or at end of period.

## Publication
- Provider review before submitting any paper or releasing any artifact that uses their data.
