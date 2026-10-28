# Solar Power Prediction, Drift Monitoring, and Manual Model Retraining

> **Report status:** Technical project report prepared from the repository and
> the GitHub Actions deployment run available on 28 October 2026. Replace the
> screenshot placeholders with your own screenshots before submitting or
> sharing the finished report. Do not include credentials, access tokens,
> private keys, or uncensored environment-variable values in screenshots.

## 1. Executive summary

This project provides a machine-learning API for predicting solar AC power,
records prediction activity in Google Cloud Storage (GCS), and monitors changes
in selected production input features. A human reviews drift results and
decides whether model retraining is appropriate. Retraining is not initiated by
drift detection.

The application uses a scikit-learn pipeline, served by FastAPI and packaged
with Docker for Google Cloud Run. Drift monitoring compares recent production
inputs with a labeled reference dataset using a two-sample
Kolmogorov–Smirnov (KS) test. Reports are written to GCS.

Model retraining is a separate, human-started local step. The retraining
pipeline creates and evaluates a candidate model. When the candidate files are
committed to `main`, the configured GitHub Actions workflow runs tests and
re-evaluates the candidate. If the quality gate passes and the protected
production environment allows deployment, the workflow builds and pushes an
image, deploys a new Cloud Run revision, and routes production traffic to it.

The latest recorded GitHub Actions run on `main` completed successfully:

- **Workflow:** Model retraining quality gate
- **Run:** [GitHub Actions run #9](https://github.com/RiteshRG/solar-drift-monitoring-api/actions/runs/40722407147)
- **Commit:** `db2a71fcda0a0df044dd2eb4a264f6a6155ac6e4`
- **Observed steps:** candidate validation, Google Cloud authentication,
  container build and push, Cloud Run deployment, and traffic routing all
  succeeded.

The candidate evaluated in the local run reported R² `0.9963766551780785`,
RMSE `523.1008687264176`, and MAE `337.5901200325454`. These matched the
repository production-baseline model's scores on the same holdout rows. The
candidate passed the required R² gate but did not demonstrate a metric
improvement.

## 2. Project objectives

The project has two related but separate operational workflows:

1. **Prediction and drift monitoring:** serve prediction requests, log their
   input and prediction records, compare recent input distributions with
   reference data, and save drift reports to GCS.
2. **Manual model retraining and deployment:** allow a person to investigate
   a drift report, explicitly run local candidate training and evaluation,
   then promote a candidate through CI/CD validation and a controlled Cloud
   Run deployment.

Drift is a signal to investigate, not an instruction to retrain. Input
distribution changes may have many causes, including weather, season, sensor
conditions, inverter operation, panel condition, or data quality. The monitor
does not identify the cause and does not measure prediction accuracy.

## 3. System architecture

```text
Training/reference data: data/train.csv
              |
              +------------------------------+
              |                              |
       Local manual training          Drift comparison
              |                       (KS test per feature)
     Candidate model + details               |
              |                       Drift report in GCS
      Local candidate evaluation              |
              |                        Human investigation
   Commit candidate files to main             |
              |                     Human decides whether
       GitHub Actions                         retraining is
       tests and re-evaluation                 appropriate
              |
       GitHub production environment
       + Workload Identity Federation
              |
       Docker image with candidate
              |
       Artifact Registry
              |
       Cloud Run new revision
              |
       Route production traffic
```

### Main components

| Component | Purpose |
|---|---|
| FastAPI (`app.py`) | Prediction, model status, drift, report, and scheduled-monitoring API endpoints. |
| scikit-learn pipeline | Scales input features and predicts `AC_POWER` with a `GradientBoostingRegressor`. |
| GCS | Stores prediction records, monitoring state, lock, and drift reports. |
| Drift monitor | Runs feature-level KS tests and constructs drift results. |
| Local retraining scripts | Train and evaluate a candidate only when explicitly run by a person. |
| GitHub Actions | Re-tests the candidate on qualifying `main` pushes and performs the configured deployment. |
| Google Workload Identity Federation (WIF) | Gives GitHub Actions short-lived Google Cloud credentials without a long-lived service-account key. |
| Artifact Registry | Stores the built, versioned container image. |
| Cloud Run | Hosts the API container and serves prediction requests. |

## 4. Model and data

### Model contract

- **Prediction target:** `AC_POWER`
- **Serving input features, in order:**
  1. `IRRADIATION`
  2. `AMBIENT_TEMPERATURE`
  3. `HOUR_SIN`
  4. `HOUR_COS`
- **Model pipeline:** `StandardScaler` followed by
  `GradientBoostingRegressor`.
- The API accepts irradiation, ambient temperature, and a timestamp. It derives
  the cyclic hour features internally before calling the model.
- The model artifact expected by the API is
  `models/model_pipeline.joblib`.

### Datasets

| Dataset or records | Use |
|---|---|
| `data/train.csv` | Labeled Plant 1 reference/training data. The retraining pipeline currently uses 469 rows. |
| `data/validation.csv` | Plant 2 data used by the production-data generation script as simulated live API input. The script sends up to its first 60 rows as prediction requests. |
| GCS `production/` objects | Recent API input and prediction records used as the current drift-monitoring window. They are not used to calculate labeled production accuracy because measured production `AC_POWER` labels are not currently recorded. |

Training and evaluation use the labeled `data/train.csv` dataset. The
retraining code creates a reproducible 80/20 split using random seed 42:

- Training partition: 375 rows.
- Holdout evaluation partition: 94 rows.
- The candidate and the repository baseline are evaluated on the same holdout
  rows.

The holdout split is reproducible and is intended to make candidate-versus-
baseline comparisons consistent. Since the sample is relatively small and the
baseline appears to have matching scores on this split, results should not be
treated as proof of real-world generalization or improvement. More independent
labeled data would strengthen future evaluation.

### Local retraining command

Using the project's Python 3.11 virtual environment and installed pinned
requirements:

```powershell
python -m scripts.retrain_model
```

This command runs training first and evaluation second. The training script
creates:

- `models/candidate_model_pipeline.joblib` — the candidate model artifact.
- `models/candidate_model_pipeline.manifest.json` — JSON details that record
  the candidate hash, training data identity, feature/target contract, split,
  dependency versions, and recorded metrics.

It does not overwrite the repository's existing
`models/model_pipeline.joblib`. A failed training or evaluation command exits
unsuccessfully. The workflow re-evaluates the exact candidate before
containerization.

## 5. Prediction API

The FastAPI application loads the production model when the application starts.
The primary prediction endpoint is `POST /predict`.

Example request:

```json
{
  "IRRADIATION": 0.8,
  "AMBIENT_TEMPERATURE": 30,
  "DATE_TIME": "10-10-2026 12:00"
}
```

Example response shape:

```json
{
  "predicted_AC_POWER": 1234.56
}
```

The numeric prediction above is illustrative; actual values depend on the
request and model. The API validates the date/time format, derives `HOUR_SIN`
and `HOUR_COS`, predicts AC power, and records the successful request and
prediction in GCS.

| Endpoint | Purpose |
|---|---|
| `GET /` | Basic API response. |
| `GET /model-status` | Indicates whether the model loaded and reports service health. |
| `POST /predict` | Produces an AC power prediction and logs its record. |
| `GET /drift` | Runs drift comparison and returns the result. |
| `POST /monitoring/check` | Authenticated endpoint intended for Cloud Scheduler. |
| `GET /drift-report` | Reads the accumulated drift report from GCS. |
| `GET /production-data` | Reads production records from GCS. |

Interactive API documentation is normally available at `/docs` on the deployed
FastAPI service.

> **Screenshot 1 — FastAPI interactive documentation**
>
> Insert a screenshot of the deployed FastAPI `/docs` page showing the available
> endpoints. Ideally show the `POST /predict` request and response schema, and
> `GET /model-status`. Do not include credentials or private tokens.
>
> `![FastAPI documentation screenshot](screenshots/01-fastapi-docs.png)`

> **Screenshot 2 — Successful prediction**
>
> Insert a screenshot of a successful `POST /predict` request and response.
> Use a harmless example request and make sure no authorization token is
> visible.
>
> `![Prediction endpoint screenshot](screenshots/02-fastapi-prediction.png)`

## 6. Production data and GCS logging

The production-data generator reads `data/validation.csv`, takes up to the
first 60 rows, creates timestamps, and sends input requests to the configured
prediction API URL. These are simulated production requests; they are not
evidence of measured power output from a live solar installation.

Successful predictions are logged as JSON objects under the GCS `production/`
prefix. Drift monitoring reads the available production JSON objects and
compares the selected input features with the reference dataset. The GCS bucket
is configured in `utils/constants.py` as
`solar-drift-monitoring-2026-868330784277`.

Important GCS object paths include:

- `production/` — prediction records used as the current feature window.
- `reports/drift_report.csv` — accumulated drift results.
- `reports/runs/` — per-run report objects.
- `monitoring/monitoring_state.json` — production-record count last monitored.
- `monitoring/monitoring.lock` — lock used to avoid concurrent monitor runs.

The scheduled monitor is intended to run after at least 50 new production
records have accumulated, and requires at least 50 current records. Its code
uses an object-generation precondition for lock acquisition and report
appending, reuses deterministic run IDs for retry handling, and releases its
lock in a `finally` block. Whether a Cloud Scheduler job is currently
configured and running must be confirmed from the Cloud Console; the repository
code alone does not prove that the schedule is active.

> **Screenshot 3 — GCS bucket contents**
>
> Insert a screenshot of the configured bucket showing representative
> `production/`, `reports/`, and `monitoring/` objects. Keep the bucket name
> visible if suitable for the report. Do not show credentials, signed URLs, or
> sensitive object contents.
>
> `![GCS bucket objects screenshot](screenshots/03-gcs-bucket.png)`

> **Screenshot 4 — Stored drift report**
>
> Insert a screenshot of `reports/drift_report.csv` or the `/drift-report`
> endpoint response. Show the feature, test, KS statistic, p-value, alpha,
> drift decision, and monitoring timestamp where possible.
>
> `![Drift report screenshot](screenshots/04-drift-report.png)`

## 7. Drift-monitoring method and interpretation

For each completed monitoring run, the monitor applies a two-sample
Kolmogorov–Smirnov test (`scipy.stats.ks_2samp`) independently to:

- `IRRADIATION`
- `AMBIENT_TEMPERATURE`

The reference distribution comes from the training data. The current
distribution comes from recent production records read from GCS. The
significance level is `alpha = 0.05`. A feature is marked as drifted when
`p_value < alpha`.

Example from a reported run:

| Feature | KS statistic | p-value | Alpha | Decision |
|---|---:|---:|---:|---|
| `IRRADIATION` | 0.1554438798 | 0.1260788411 | 0.05 | No drift flagged |
| `AMBIENT_TEMPERATURE` | 0.4036728798 | 0.0000000150 | 0.05 | Drift flagged |

Interpretation:

- The **KS statistic** is the largest distance between the empirical
  cumulative distributions of the reference and current samples. A larger
  statistic indicates a larger observed distribution difference.
- The **p-value** measures how compatible the observed difference is with the
  null hypothesis that both samples come from the same distribution. A very
  small p-value is evidence against that hypothesis.
- **Alpha** is the chosen statistical decision threshold. Here, `0.05` means
  the monitor flags the feature when its p-value is below 0.05.
- A drift flag identifies a feature distribution change. It does **not** show
  that model accuracy has fallen, prove a cause, or automatically justify
  retraining.

This implementation tests two features separately and does not currently
apply a multiple-comparison correction. Interpret the p-values as monitoring
signals and investigate the data and operating context before acting.

## 8. Manual candidate evaluation and quality gate

The candidate evaluator verifies the recorded candidate and data details,
checks the exact feature and target contract, confirms dependency/runtime
compatibility, loads both candidate and baseline, and recalculates metrics on
the fixed holdout set.

The required automatic quality gate is:

```text
R² must be strictly greater than 0.80
```

Exactly `0.80` fails. RMSE and MAE are calculated and reported, but there are
no configured RMSE/MAE pass/fail limits. Their interpretation remains part of
human review.

Latest recorded local evaluation:

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| Candidate | 0.9963766552 | 523.1008687 | 337.5901200 |
| Repository baseline | 0.9963766552 | 523.1008687 | 337.5901200 |
| Candidate minus baseline | 0.0000000000 | 0.0000000000 | 0.0000000000 |

The candidate passed the strict R² gate, but the equal results show no
measured improvement on this evaluation split. Evaluation data is from the
labeled reference dataset, not from GCS production logs, because current
production logs do not include measured AC-power labels.

## 9. GitHub Actions CI/CD and Cloud deployment

### Trigger and validation

The current workflow is
`.github/workflows/model-retraining.yml`. It is configured to run on a push to
`main` when files matching the workflow's path filters change, including the
candidate model and its JSON details file, training data, serving code,
training/evaluation code, tests, Dockerfile, and relevant configuration.
Unrelated pushes outside those filters do not start this workflow.

The validation job uses GitHub-hosted Ubuntu runners and Python 3.11. It
installs the pinned project requirements, runs the model-contract unit tests,
and re-evaluates the candidate. It exports the candidate SHA-256 so the
deployment job can verify it is packaging the same artifact that passed
validation.

### Authentication

The deployment job uses GitHub's OIDC token with Google Workload Identity
Federation. Google exchanges that short-lived identity for credentials
associated with the deployment service account. This avoids placing a
long-lived Google service-account JSON key in the repository or GitHub.

The workflow reads these GitHub `production` environment variables:

- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_DEPLOY_SERVICE_ACCOUNT`
- `GCP_PROJECT_ID`
- `GCP_REGION`
- `CLOUD_RUN_SERVICE`
- `ARTIFACT_REGISTRY_REPOSITORY`

The WIF provider trust policy and Google IAM grants must be configured for the
intended repository, `main` branch, production environment, and deployment
operations. The GitHub `production` environment can also require a human
reviewer before deployment proceeds. The Cloud Run runtime identity, used by
the running API for its GCS access, is separate from the GitHub deploy
identity.

### Image and Cloud Run release

After validation and production-environment approval, the deployment job:

1. Re-evaluates the candidate and checks its SHA-256 against the validated
   candidate.
2. Builds a Docker image from the repository.
3. Uses the candidate artifact as the image's
   `models/model_pipeline.joblib` serving model path. This preserves the
   existing API loading contract; it does not rename or overwrite the model in
   the developer's local repository.
4. Tags the image with the commit SHA and pushes it to the configured
   Artifact Registry repository.
5. Deploys a new Cloud Run revision with no traffic initially, using a short
   revision tag based on the commit.
6. Routes production traffic to the new revision.

The current workflow version does **not** run a smoke test between deployment
and traffic routing. As a result, a successful deployment command and traffic
update do not independently prove the prediction endpoint works after release.
The project should retain the previous known-good Cloud Run revision so an
operator can route traffic back if a problem is observed.

### Verified deployment evidence

GitHub Actions run #9 completed successfully on 28 October 2026 for commit
`db2a71fcda0a0df044dd2eb4a264f6a6155ac6e4`. The recorded job steps confirm:

- Candidate tests and evaluation succeeded.
- Deployment configuration validation succeeded.
- WIF authentication and `gcloud` setup succeeded.
- Docker image build and Artifact Registry push succeeded.
- Cloud Run deployment succeeded.
- Production traffic routing succeeded.

This confirms the workflow completed those steps at the time of that run. Add
Cloud Console screenshots below to visually document the resulting service,
revision, image, and traffic allocation.

> **Screenshot 5 — GitHub Actions workflow summary**
>
> Insert a screenshot of the successful run showing both
> `validate-candidate` and `deploy-approved-candidate` as successful. The run
> used for the evidence above is [run #9](https://github.com/RiteshRG/solar-drift-monitoring-api/actions/runs/40722407147).
>
> `![GitHub Actions run summary](screenshots/05-actions-summary.png)`

> **Screenshot 6 — GitHub Actions deployment steps**
>
> Insert a screenshot of the job steps showing Google authentication, image
> build/push, Cloud Run deployment, and traffic routing. Do not include tokens
> or credential contents.
>
> `![GitHub Actions deployment steps](screenshots/06-actions-deploy-steps.png)`

> **Screenshot 7 — Artifact Registry image**
>
> Insert a screenshot from Google Cloud Artifact Registry showing the
> `solar-monitoring` repository and the image/tag associated with the
> successful deployment commit. Preferably show the image digest as well.
>
> `![Artifact Registry image](screenshots/07-artifact-registry.png)`

> **Screenshot 8 — Cloud Run service and revision**
>
> Insert a screenshot of the `solar-monitoring-api` Cloud Run service in
> `asia-south1`, showing the current revision and traffic allocation. Include
> the service URL only if it is public information appropriate for your
> submission.
>
> `![Cloud Run service and revision](screenshots/08-cloud-run-service.png)`

> **Screenshot 9 — GitHub production environment**
>
> Insert a screenshot showing the `production` environment and its approval
> protection, if configured. Hide or blur environment variable values and any
> sensitive identifiers not needed in the report.
>
> `![GitHub production environment](screenshots/09-github-production-environment.png)`

> **Screenshot 10 — Workload Identity Federation configuration**
>
> Insert a screenshot showing the WIF pool/provider and its trust configuration
> at a high level. Do not show credentials, tokens, or private key material.
>
> `![Google Workload Identity Federation](screenshots/10-google-wif.png)`

## 10. Current deployment configuration

The repository constants identify the configured Google Cloud project as
`solar-drift-monitoring-2026`, the region as `asia-south1`, the Cloud Run
service as `solar-monitoring-api`, the Artifact Registry repository as
`solar-monitoring`, and the GCS bucket as
`solar-drift-monitoring-2026-868330784277`.

GitHub Actions deployment values are stored as environment variables in the
GitHub `production` environment. Do not copy credentials or tokens into this
report. Verify the Cloud Console values against the screenshots before
submitting, because cloud resources and configuration can change independently
of repository code.

> **Screenshot 11 — Production-data generation**
>
> Insert a screenshot of the terminal running
> `scripts/generate_production_data.py`, showing successful API response
> statuses. Exclude any credentials or private request headers.
>
> `![Production data generation](screenshots/11-production-data-generation.png)`

> **Screenshot 12 — Local retraining and evaluation**
>
> Insert a screenshot of the terminal running
> `python -m scripts.retrain_model` in the Python 3.11 environment. Show the
> candidate metrics and quality-gate result, but avoid unrelated local paths
> or personal information if the report will be shared publicly.
>
> `![Local retraining and evaluation](screenshots/12-local-retraining.png)`

## 11. Security, reliability, and limitations

### Security and access

- GitHub Actions uses OIDC/WIF rather than a long-lived service-account JSON
  key.
- Deployment permissions should be least-privilege and restricted to the
  intended project, service, and repository trust conditions.
- Keep GitHub production-environment approvals enabled if human deployment
  approval is part of the project process.
- Do not place API keys, access tokens, private keys, passwords, or unredacted
  secret values in screenshots or report files.
- The Cloud Run runtime service account and the GitHub deployment service
  account have different responsibilities.

### Reliability and model-quality limitations

- Drift detection is an input-distribution check, not an accuracy metric.
- The current production records contain predictions but do not include
  verified measured AC-power labels, so production error metrics cannot
  currently be calculated.
- The 469-row reference dataset is limited in size. The fixed holdout supports
  repeatable comparison but does not replace evaluation on a newer,
  independent dataset.
- The current candidate's metrics match the existing baseline. Passing the
  threshold does not mean the candidate is better.
- RMSE and MAE are reported but have no automated acceptance thresholds.
- The current deployment workflow switches traffic without a smoke test.
  Keep the previous revision available and verify the service after deployment.
- The scheduled monitoring route is implemented, but a working Cloud
  Scheduler job must be confirmed separately in Google Cloud.
- The example generation script and historical drift output are project test
  evidence; do not describe them as measurements from a live solar plant
  unless independent operational evidence confirms that.

## 12. Suggested evidence and screenshot checklist

Before creating the final formatted report, gather:

- [ ] FastAPI `/docs` page and a successful prediction response.
- [ ] Drift response with KS statistic, p-value, alpha, and feature decision.
- [ ] GCS bucket objects showing production records, reports, and monitoring
  state.
- [ ] Local `scripts.retrain_model` output showing candidate metrics.
- [ ] GitHub Actions successful workflow summary and deployment job details.
- [ ] Artifact Registry image/tag and image digest.
- [ ] Cloud Run service, deployed revision, and traffic allocation.
- [ ] GitHub `production` environment protection screen, with values hidden.
- [ ] Google Workload Identity Federation configuration screen, with sensitive
  values hidden.
- [ ] Optional Cloud Scheduler job details if the scheduled monitor is
  configured and running.

Use descriptive screenshot filenames matching the placeholders above, or edit
the image paths after inserting your screenshots. Add a short caption below
each image explaining what it demonstrates and which workflow stage it
supports.

## 13. Prompt for a cloud-based report-generation AI

Copy the prompt below into your report-generation AI after providing this
Markdown file and the screenshots:

```text
Create a polished, detailed technical project report using the supplied
"detail report.md" as the factual source and the supplied screenshots as
evidence.

Requirements:
1. Preserve the report's technical meaning, values, caveats, and distinction
   between drift detection, model accuracy evaluation, and deployment.
2. Insert each supplied screenshot at the matching labeled screenshot
   placeholder. Use the screenshot's actual contents; do not invent or alter
   cloud statuses, metrics, URLs, image tags, revision names, or configuration.
3. Add a concise figure number and caption for every screenshot. Explain what
   each screenshot demonstrates in the workflow.
4. If a placeholder has no matching screenshot, leave a clearly marked
   "Screenshot not supplied" placeholder. Do not fabricate an image or claim
   that a cloud resource is configured based only on repository code.
5. Keep the report understandable to a technical project evaluator. Explain
   the KS statistic, p-value, alpha threshold, R² gate, candidate model,
   GitHub Actions, Workload Identity Federation, Artifact Registry, and Cloud
   Run in plain language while retaining the correct technical details.
6. Clearly state that drift is not proof of reduced prediction accuracy, that
   the candidate metrics matched the baseline in the recorded evaluation, and
   that production accuracy is not measured without labeled production
   outcomes.
7. Preserve security: redact API keys, access tokens, private keys, passwords,
   secret values, and any sensitive account information visible in screenshots.
   Never reproduce hidden or redacted values.
8. Use a professional academic report structure with a title page, executive
   summary, architecture, methodology, implementation, CI/CD and deployment,
   results, limitations, conclusion, and figure captions. Add a table of
   contents if the output format supports it.
9. Do not claim that the current deployment has a smoke test; the documented
   workflow routes traffic without one. Mention this as a limitation.
10. Deliver the final report as a polished document. Keep the supplied
    Markdown as the source of truth; if any screenshot appears to conflict
    with the text, identify the discrepancy instead of silently choosing a
    value.
```

## 14. Conclusion

The project integrates solar AC-power prediction, input drift monitoring,
human-led candidate retraining, and GitHub Actions deployment to Google Cloud.
Its separation of concerns is important: monitoring produces evidence for
human investigation, local scripts create and evaluate candidates, and CI/CD
validates and deploys a specific candidate only after the configured
production gate.

The recorded GitHub Actions run demonstrates that candidate validation,
Google Cloud authentication, image publication, Cloud Run deployment, and
traffic routing completed successfully for the recorded commit. The candidate
passed the configured R² gate but matched the existing baseline's metrics.
Future model-quality improvements should prioritize collecting labeled
production outcomes and evaluating candidates on sufficiently large,
independent, representative data.
