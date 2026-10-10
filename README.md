# Solar Power Model Monitoring

An end-to-end solar AC-power prediction and monitoring project. A scikit-learn Gradient Boosting model is served through FastAPI, packaged with Docker, and documented for deployment to Google Cloud Run. Prediction requests are logged to Cloud Storage (GCS), and a SciPy two-sample Kolmogorov–Smirnov (KS) test compares recent production inputs with a training-data reference window.

> **Academic project:** See [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md) for the detailed report, experiment interpretation, evidence index, and assignment compliance table. Some GCP and screenshot details in that report are identified as unverified because the corresponding image files/live cloud configuration were not available during report preparation.

## What it does

- Predicts `AC_POWER` from irradiation, ambient temperature, and hour-of-day features.
- Serves predictions through FastAPI.
- Logs successful prediction inputs and outputs to GCS.
- Compares `IRRADIATION` and `AMBIENT_TEMPERATURE` in the training reference data and current GCS production data using the KS test at `alpha = 0.05`.
- Provides a load-test script configured for 100 requests and 10 concurrent workers; it calculates client-observed p50, p95, and p99 for successful responses.

## Architecture

```text
Solar generation and weather data
              |
        Preprocessing
              |
 Gradient Boosting pipeline
              |
           FastAPI
              |
       Docker container
              |
       Artifact Registry
              |
          Cloud Run
          /       \
 Cloud Monitoring  Cloud Storage
                   /          \
          production logs   drift report
                  \            /
             KS drift comparison
      (data/train.csv vs recent GCS data)
```

The repository includes a helper that checks whether 50 new production records have accumulated. It does **not** contain a Cloud Scheduler job definition, so an active scheduled monitoring job should not be assumed.

## Model and data

The model artifact is `models/model_pipeline.joblib`, a scikit-learn pipeline with `StandardScaler` and `GradientBoostingRegressor(random_state=42)`.

| Purpose | Data |
|---|---|
| Target | `AC_POWER` |
| Input features | `IRRADIATION`, `AMBIENT_TEMPERATURE`, `HOUR_SIN`, `HOUR_COS` |
| Reference data | `data/train.csv` (Plant 1 data; 469 rows) |
| Validation data | `data/validation.csv` (Plant 2 data; 513 rows) |
| Current production window | Recent JSON records from the GCS `production/` prefix |

The notebook merges generation and weather records by `DATE_TIME`, aggregates to hourly records, removes records with no irradiation, and encodes hour cyclically using sine and cosine. The production-data generator posts validation rows to the prediction API, providing a Plant 1 versus Plant 2 domain-shift scenario.

The notebook’s saved evaluation results for Gradient Boosting are RMSE `523.1008687264176` and R² `0.9963766551780785` on its held-out split; its saved five-fold cross-validation means are RMSE `627.613143` and R² `0.993689`. These are historical notebook outputs, not a fresh training run.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Basic API message |
| `GET` | `/model-status` | Model load/health status |
| `POST` | `/predict` | Predict AC power and log the record to GCS |
| `GET` | `/drift` | Compare reference and current windows; save and return the latest drift results |
| `GET` | `/drift-report` | Read the accumulated drift report from GCS |
| `GET` | `/production-data` | Read production JSON records from GCS |

Example request:

```json
{
  "IRRADIATION": 0.8,
  "AMBIENT_TEMPERATURE": 30,
  "DATE_TIME": "10-10-2026 12:00"
}
```

The timestamp format is `DD-MM-YYYY HH:MM`. The response contains `predicted_AC_POWER`. The request schema constrains irradiation to 0–2.0 and ambient temperature to -10–100.

## Drift monitoring

The monitor uses the training dataset as the reference and the latest production records from GCS as the current window. It requires at least 50 current records, tests `IRRADIATION` and `AMBIENT_TEMPERATURE` independently, and flags drift when `p_value < 0.05`. The report is appended to `reports/drift_report.csv` in the configured GCS bucket.

The latest run in the supplied local drift CSV is timestamped `2026-10-10T14:48:39.132124`:

| Feature | KS statistic | p-value | Alpha | Drift detected |
|---|---:|---:|---:|---|
| `IRRADIATION` | 0.15544397826535525 | 0.12607884106919445 | 0.05 | No |
| `AMBIENT_TEMPERATURE` | 0.4036728798404292 | 1.5044186442025674e-08 | 0.05 | Yes |

This indicates a statistically significant feature-distribution difference for ambient temperature in that run. It does **not**, by itself, show that prediction quality has degraded. Quality monitoring requires true production AC-power outcomes, which are not currently logged.

## Local development

Requirements: Python 3.11 (the Docker image uses `python:3.11-slim`), pip, and Google Cloud credentials with appropriate access to the configured GCS bucket.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

The application reads a model artifact from `models/model_pipeline.joblib` and uses GCS for prediction logging and drift data. For local development, configure Google Application Default Credentials in your environment:

```bash
gcloud auth application-default login
```

Run the API locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

Open the interactive API docs at `http://localhost:8080/docs`.

### Run project scripts

```bash
python scripts/generate_production_data.py
python scripts/load_test.py
python monitoring/monitoring_trigger.py
```

- `generate_production_data.py` posts up to 60 rows from `data/validation.csv` to the Cloud Run URL configured in the script. It is separate from the load test.
- `load_test.py` sends 100 requests with 10 concurrent workers and prints successful/failed request counts and p50/p95/p99 latency. The script currently targets the documented Cloud Run URL; inspect `API_URL` before use.
- `monitoring_trigger.py` checks the count of GCS production objects against the last stored count. It does not itself invoke `/drift` or configure Cloud Scheduler.

The load-test script does not remove or separately identify cold-start requests. Account for a possible Cloud Run cold start when interpreting the percentiles.

## Docker

Build and run the image locally:

```bash
docker build -t solar-monitoring-api .
docker run --rm -p 8080:8080 solar-monitoring-api
```

The container listens on port `8080`. Cloud Storage access must be configured securely for the runtime; do not copy credential files into the image.

## Documented Google Cloud deployment

The project context documents these resource identifiers; confirm their current state before deploying:

| Resource | Documented value |
|---|---|
| Project ID | `solar-power-monitoring-511206` |
| Region | `asia-south1` |
| Cloud Run service | `solar-monitoring-api` |
| Artifact Registry repository | `solar-monitoring` |
| GCS bucket | `solar-monitoring-511206` |
| Image | `asia-south1-docker.pkg.dev/solar-power-monitoring-511206/solar-monitoring/solar-monitoring-api:latest` |

Build, authenticate, tag, and push:

```bash
docker build -t solar-monitoring-api .
gcloud auth configure-docker asia-south1-docker.pkg.dev
docker tag solar-monitoring-api:latest asia-south1-docker.pkg.dev/solar-power-monitoring-511206/solar-monitoring/solar-monitoring-api:latest
docker push asia-south1-docker.pkg.dev/solar-power-monitoring-511206/solar-monitoring/solar-monitoring-api:latest
```

Deploy to Cloud Run:

```bash
gcloud run deploy solar-monitoring-api \
  --image=asia-south1-docker.pkg.dev/solar-power-monitoring-511206/solar-monitoring/solar-monitoring-api:latest \
  --region=asia-south1 \
  --platform=managed \
  --allow-unauthenticated \
  --service-account=solar-monitoring-run@solar-power-monitoring-511206.iam.gserviceaccount.com
```

`--allow-unauthenticated` exposes the service publicly. Use appropriate authentication and access controls for anything beyond a controlled academic demonstration. The command values above are documented project context, not confirmation that the service or its IAM configuration is currently active.

## Monitoring and evidence

Google Cloud Monitoring is identified in the project context for Cloud Run request count and latency metrics. The repository’s load-test script computes p50/p95/p99, but the actual captured output was not available when this README was prepared. Previously noted Cloud Monitoring percentile values are documented in [the final report](FINAL_PROJECT_REPORT.md) as historical, unverified observations and should not be represented as the measured output of the 100-request test.

Screenshot filenames were supplied for Cloud Run, Artifact Registry, request count, latency, endpoints, production records, drift results, and a budget alert. The image files were not available in the repository, so their contents have not been independently verified. See the report’s evidence index for the mapping and verification caveats.

## Security and cost

- Never commit API keys, passwords, tokens, ADC files, or service-account key files.
- Keep local secrets and credentials out of Docker images and submitted archives.
- The documented budget alert is a notification, not a spending cap; check billing and remaining credits regularly.
- After evaluation, preserve required evidence and reports, then review and clean up unused Cloud Run services, container images, storage buckets, Scheduler jobs, VMs, and other billable resources.

## Repository layout

```text
.
├── app.py
├── Dockerfile
├── requirements.txt
├── data/
├── models/
├── monitoring/
├── notebooks/
├── reports/
├── schema/
├── scripts/
└── utils/
```

Local environments, caches, credentials, and `backup/` are not part of the production architecture.
