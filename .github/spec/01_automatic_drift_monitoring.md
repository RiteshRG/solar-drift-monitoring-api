# Automatic Drift Monitoring

## Purpose

Use drift reports to prompt human investigation. Drift detection must not start
training or deployment automatically.

## Automatic drift monitoring

- Cloud Scheduler calls the protected `POST /monitoring/check` endpoint every
  five minutes using an OIDC token from a dedicated service account.
- When at least 50 new production predictions have arrived since the previous
  completed run, the endpoint invokes drift monitoring. The scheduled endpoint
  validates the token audience and expected service-account identity.
- Run drift checks separately from prediction requests so prediction latency
  and availability do not depend on monitoring.
- Save each completed run with a unique run ID and append its feature results
  to the accumulated GCS drift CSV. The current report records one row per
  feature, so a run may add multiple CSV rows.
- Use a GCS lock and deterministic run ID to prevent overlapping runs and
  duplicate report rows.
- Advance the last-monitored prediction count only after the drift report is
  successfully persisted.
- Email notifications are out of scope. A human reviews the GCS report and
  decides whether retraining is justified.

## Workflow

1. The monitoring service checks automatically after each 50 new predictions.
   It persists a per-run report and appends the feature results to the GCS CSV.
2. A human reviews the report and investigates any detected drift.
3. If retraining is justified, follow the separate specification in
   `02_manual_model_retraining.md`.
