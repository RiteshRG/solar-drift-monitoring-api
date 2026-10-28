# Automatic Drift Monitoring Implementation Plan

## Goal

Run drift monitoring automatically after at least 50 new prediction records
have accumulated, append results to the GCS drift CSV, and retain an immutable
report for each run. Email, retraining, and model deployment are out of scope.

## Implementation steps

1. **Protect the scheduled endpoint**
   - Expose `POST /monitoring/check` for Cloud Scheduler.
   - Validate the Google OIDC token audience and the configured scheduler
     service-account identity.
   - Keep this endpoint separate from prediction requests.

2. **Configure periodic threshold checks**
   - Configure Cloud Scheduler to call the endpoint every five minutes.
   - Run drift monitoring only when at least 50 new GCS production records
     exist since the last successful monitoring run.
   - Return a no-op status below the threshold.

3. **Make monitoring safe to retry**
   - Use a GCS lock to prevent overlapping runs.
   - Derive a deterministic run ID from the last monitored count and current
     production count.
   - Write a per-run CSV under `reports/runs/` and append feature rows to the
     accumulated `reports/drift_report.csv` without duplicating a run.
   - Update the last-monitored count only after report persistence succeeds.

4. **Verify behavior**
   - Test unauthenticated and invalid-token calls are rejected.
   - Test under-threshold checks do not create reports or advance state.
   - Test a due monitoring run writes a unique report and appends exactly one
     row per monitored feature.
   - Test retries and overlapping calls do not duplicate report rows.
   - Confirm drift status is returned to the scheduler caller. Do not send
     emails, retrain models, or deploy models from this flow.

## Cloud configuration

- Deploy the API with the scheduler identity and OIDC audience configured.
- Create a Cloud Scheduler job using the dedicated scheduler service account.
- Grant only the permissions needed for scheduler invocation and GCS report/state
  access.
- Confirm bucket IAM for the Cloud Run runtime identity before enabling the job.

## Out of scope

- Email or other alert delivery.
- Model training, evaluation thresholds, CI/CD, and model promotion.
