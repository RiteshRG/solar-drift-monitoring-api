# Manual Model Retraining

## 1. Purpose and scope

Define an end-to-end, human-approved workflow for retraining the solar AC-power
regression model using the repository's labeled `data/train.csv`.

The workflow starts only after a person reviews a drift report and decides that
retraining is justified. Drift detection, report creation, and scheduled
monitoring must never start model training, open a model pull request, pass a
quality gate, or deploy a model automatically.

This specification covers local training, candidate evaluation, GitHub
pull-request validation, and reviewed promotion to Cloud Run. Email
notifications are out of scope.

## 2. Current project facts to preserve

- Production inference is served by FastAPI in `app.py`.
- The API loads the production model at process startup from
  `PRODUCTION_MODEL_PATH` in `utils/constants.py`.
- The current checked-in artifact is `models/model_pipeline.joblib`.
- The current model is a scikit-learn pipeline containing `StandardScaler`
  and `GradientBoostingRegressor(random_state=42)`.
- Training data is `data/train.csv`, with 469 rows and columns:
  `AC_POWER`, `IRRADIATION`, `AMBIENT_TEMPERATURE`, `HOUR_SIN`, and `HOUR_COS`.
- The feature order and names used for prediction are
  `IRRADIATION`, `AMBIENT_TEMPERATURE`, `HOUR_SIN`, and `HOUR_COS`.
- `AC_POWER` is the supervised target. Do not use it as an inference feature.
- The API's request schema contains irradiation, ambient temperature, and
  date/time; the serving code derives `HOUR_SIN` and `HOUR_COS`. Candidate
  artifacts must remain compatible with this contract.
- Existing production logs contain model inputs and predicted AC power, not
  measured AC-power outcomes. They are not ground-truth labels and must not be
  used to claim production accuracy.
- Drift statistics, including KS-test p-values, describe feature
  distributions. They are not model-performance metrics.

The values above describe the current checkout. Before implementation after an
account switch, re-check the model, data, constants, deployment service, and
artifact paths rather than assuming old cloud resources or account settings
still apply.

## 3. Required human-controlled workflow

1. A human reviews the GCS drift report and investigates possible causes such
   as sensors, inverter behavior, panels, weather, seasonality, or data quality.
2. The human explicitly decides whether retraining should be attempted.
3. The human invokes a local training command. No API endpoint, Cloud Scheduler
   job, prediction request, drift report, or GitHub push may start training.
4. Local training validates the approved labeled dataset and writes a candidate
   artifact to `CANDIDATE_MODEL_PATH`. It must not modify or overwrite
   `PRODUCTION_MODEL_PATH`.
5. The training run writes a machine-readable candidate manifest next to the
   candidate artifact.
6. The human reviews the candidate metrics and its recorded details, then
   commits the candidate and required metadata directly to `main`.
7. GitHub Actions runs for model-related pushes to `main`, validates the
   candidate against the fixed evaluation data and production baseline, and
   stops on a failed check.
8. A passing check is necessary but not sufficient for deployment. A
   designated reviewer approves the protected production deployment. Deploy
   only from the protected `main` branch after the quality workflow passes.
9. The deployment job packages and deploys the exact candidate artifact
   validated by CI as a new Cloud Run revision.
10. Route production traffic to the new revision without a smoke test. Keep
    the prior revision available and document how to route traffic back to it.

## 4. Data preparation and split requirements

- Use `TRAINING_DATA_PATH` as the default labeled training source. Do not
  silently fetch or mix in production records.
- Validate the CSV before fitting: required columns exist exactly once, values
  are numeric and finite, and no required feature or target is missing.
  Invalid data must stop training with a clear error.
- Preserve the current feature schema and cyclic hour features. If features or
  preprocessing change, update training, evaluation, and serving compatibility
  tests together.
- With the current 469-row dataset, use the initial reproducible split: 80%
  training and 20% holdout, `shuffle=True`, and `random_state=42`. Preserve the
  evaluation rows from candidate fitting and include their row indices and a
  split identity hash in the candidate manifest.
- Require at least two rows in both training and evaluation partitions so R²
  is defined. Report exact counts; this mathematical minimum alone does not
  imply the sample is statistically robust.
- Fit all learned preprocessing only on the training partition. Evaluate both
  production and candidate models on the same untouched holdout rows.
- Do not tune a candidate repeatedly against the final holdout and then treat
  its score as an unbiased final estimate. If the holdout is used for model
  selection, document that limitation and prefer cross-validation or a
  separately governed final evaluation set.
- `data/validation.csv` represents a different plant according to the project
  documentation. Keep it as a separate domain-shift evaluation unless a
  domain owner explicitly approves its use. Do not automatically mix it into
  training or combine its metrics with the primary holdout.
- The current data is small, so report the sample count and interpret scores
  cautiously. Do not claim that a single split proves future seasonal or
  cross-plant performance.

## 5. Training behavior and candidate outputs

Add a local, explicitly invoked training script, expected at
`scripts/train_model.py`.

The script must:

- Read the configured training CSV and validate it before fitting.
- Use the four approved input features and `AC_POWER` target.
- Initially reproduce the serving model family and preprocessing:
  `StandardScaler` followed by `GradientBoostingRegressor`, with fixed
  reproducible settings.
- Fit only on the training partition; do not fit on the holdout evaluation
  rows.
- Save the resulting pipeline only to `CANDIDATE_MODEL_PATH`.
- Never overwrite, rename, or promote the production model as a side effect.
- Exit unsuccessfully on invalid input, fitting failure, or failed artifact
  serialization. Do not produce success-shaped fallback artifacts.
- Produce a candidate manifest containing at least:
  - candidate artifact path and SHA-256;
  - training data path and SHA-256 (or immutable dataset version);
  - training code Git commit, if available;
  - training and evaluation row counts;
  - feature names/order and target name;
  - model/preprocessing type and hyperparameters;
  - random seed and split definition;
  - Python, scikit-learn, NumPy, pandas, and joblib versions;
  - evaluation metric values and the quality-gate result.

The candidate artifact and manifest form one candidate. Do not hand-edit
metrics or manifest values independently of the artifact.
Training must use the Python and scikit-learn versions pinned for the
production container. A runtime version mismatch must stop training.

## 6. Evaluation and quality gate

Add an evaluation command, expected at `scripts/evaluate_model.py`, that loads
the candidate and the production baseline and evaluates both on identical,
labeled holdout rows.

### Mandatory R² gate

- Compute R² using scikit-learn's regression metric against actual
  `AC_POWER` labels in the agreed evaluation set.
- The candidate passes the R² requirement only if **R² is strictly greater
  than 0.80** (`R² > 0.80`). A score equal to `0.80` fails.
- A non-finite score, missing score, invalid target/prediction data, or
  insufficient evaluation data fails closed.
- The threshold applies to the agreed holdout evaluation, not training-set
  score, a drift p-value, or model-reported metadata alone.

### Required metric reporting and comparison

- Report R², RMSE, and MAE for both production baseline and candidate using the
  same evaluation rows. RMSE and MAE are expressed in AC-power units.
- R² > 0.80 is the explicitly agreed minimum gate. It does not by itself
  guarantee acceptable absolute power error. Before enabling automatic
  promotion, agree on acceptable RMSE/MAE limits or a non-regression tolerance
  relative to the baseline.
- Until RMSE/MAE limits are approved, CI must display these metrics and must
  not silently treat an unspecified error threshold as passed. Deployment
  remains subject to a human reviewer.
- Do not use historical notebook metrics as the baseline without recomputing
  them under the selected data split and current dependency versions.

### Evaluation checks

CI must fail when any of the following occurs:

- training/evaluation data validation fails;
- a required column is missing, has invalid values, or has an unexpected
  schema;
- candidate or baseline cannot be loaded;
- prediction output has the wrong shape or contains non-finite values;
- metrics cannot be computed or are non-finite;
- the candidate R² is not strictly greater than 0.80;
- candidate metadata does not match the candidate artifact hash or evaluation
  run;
- model interface/contract tests fail.

The gate decision must include metrics, sample counts, data/split identity,
candidate hash, and a clear PASS/FAIL reason in the GitHub Actions output.

## 7. Candidate storage and GitHub push

- Commit candidate artifacts and their recorded details directly to `main`
  for the configured workflow. Check artifact size and repository growth
  before choosing ordinary Git storage.
- If the artifact is too large or repository history growth becomes a concern,
  use Git LFS or a versioned artifact store. Do not use mutable `latest`
  objects for candidate identity.
- Include the candidate manifest, evaluation output, and a human explanation
  of why the drift investigation justified retraining.
- CI must recompute evaluation metrics; it must not trust locally supplied
  metric values as proof that the candidate passed.
- CI must verify the candidate SHA-256 and associate the check result with
  that exact hash.
- A failed gate must stop the deployment job.
- Treat joblib model artifacts as executable pickle inputs. Run evaluation
  only on isolated ephemeral GitHub-hosted runners.

## 8. CI/CD and production deployment

- Validation and deployment run only for model-related pushes to `main`.
- Use GitHub Actions OIDC with Google Cloud Workload Identity Federation for
  short-lived credentials. Do not create or store a long-lived Google service
  account JSON key in GitHub Secrets.
- Configure a protected GitHub production environment with required human
  reviewers. Passing CI alone must not deploy the candidate.
- Configure production environment variables for the Workload Identity
  Federation provider, deploy service account, project, region, Cloud Run
  service, and Artifact Registry repository. Do not store a service-account
  private key as a GitHub secret.
- Grant the deploy identity only the permissions needed to publish an image
  and update the intended Cloud Run service.
- Build an immutable, versioned container image that contains the candidate
  artifact whose hash passed CI. Record the image digest and candidate hash.
- Do not rebuild using a different model between evaluation and deployment.
- Deploy as a new Cloud Run revision; do not replace the artifact inside a
  running revision or mutate a previously deployed image tag.
- Keep the prior known-good revision available. This project workflow routes
  traffic without smoke-testing the candidate; document how to route traffic
  back to the prior revision if problems occur.
- The Cloud Run runtime service account and its GCS access are separate from
  the GitHub deploy identity. Do not conflate their permissions.

## 9. New Google account/project readiness

After switching Google accounts, first confirm the active account and selected
project. Do not assume the previous project resources, permissions, URL,
bucket, service accounts, or billing configuration are available in the new
account.

Before deploying through CI:

- confirm the destination project, billing, region, Cloud Run service, Artifact
  Registry repository, GCS bucket, and current production model;
- update project-specific configuration without committing credentials;
- confirm Cloud Run can read/write the required GCS objects using its runtime
  service account;
- configure Workload Identity Federation between the GitHub repository and
  Google Cloud, and verify the trust conditions restrict the intended
  repository, branch, and environment;
- grant the GitHub deploy service account only the necessary Artifact Registry
  and Cloud Run deployment permissions;
- configure GitHub environment protection and required reviewers;
- verify that the deployed baseline still serves the expected request contract
  before introducing a candidate model.

Never put Google account passwords, access tokens, service-account private
keys, or Brevo credentials in source code, the candidate manifest, or a GitHub
workflow file.

## 10. Tests and acceptance criteria

Implementation is ready for review when all of the following are true:

1. A person can explicitly train a candidate locally from the approved
   `data/train.csv` workflow.
2. Training leaves `PRODUCTION_MODEL_PATH` unchanged and writes the candidate
   manifest and artifact hash.
3. Candidate and production baseline are evaluated on the same fixed labeled
   evaluation rows.
4. R² > 0.80 is required; R² <= 0.80 fails the gate.
5. RMSE and MAE are reported, and their production acceptance policy is
   documented before unattended promotion is enabled.
6. Model interface tests verify the API-derived features and prediction output
   remain compatible.
7. A failed or invalid candidate stops before deployment.
8. A passing candidate still requires explicit production environment approval.
9. The deployed image contains the exact candidate hash validated by CI.
10. Drift monitoring only writes reports; it never trains, promotes, or
    deploys a model.
11. The prior Cloud Run revision remains available for rollback.
12. Validation runs only for pushes to `main`; production deployment requires
    protected-environment approval.
