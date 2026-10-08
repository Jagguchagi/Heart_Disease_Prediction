# Heart Disease MLOps Assignment

A reproducible implementation of the MLOps assignment using the UCI Heart Disease Cleveland subset. Repository: https://github.com/Jagguchagi/Heart_Disease_Prediction. This is an educational demonstration, not a clinical decision tool.

For Linux lab VM setup, missing GitHub files, Docker, GitHub Actions, Kubernetes, and submission evidence, follow [LAB_VM_GUIDE.md](LAB_VM_GUIDE.md). The published repository currently uses a flat layout while this local checkout nests the application under `Heart_Disease/`; verify the upload is complete before cloning.

## Current implementation

- Reads the original headerless UCI Cleveland file and converts its severity label `num` (0-4) to a binary `target` (0 = absent; 1 = any presence).
- Handles the six source missing values with training-fold median/mode imputation, scales continuous inputs, and one-hot encodes coded categories.
- Generates class balance, continuous feature histograms, numeric correlation heatmap, and a JSON summary.
- Tunes Logistic Regression and Random Forest with stratified cross-validation; compares accuracy, precision, recall, and ROC-AUC; logs local MLflow runs to SQLite.
- Saves the selected complete scikit-learn pipeline as `models/heart_disease_pipeline.joblib`.
- Serves typed requests using FastAPI, exposes `/health`, `/predict`, and `/metrics`, and avoids logging patient feature values.
- Includes tests, a local Jenkins pipeline, Docker and Compose files, and Kubernetes manifests.

## Setup on Windows

Use Python 3.10 and run these commands from this folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your local execution policy blocks environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

## Data source

The project uses the processed Cleveland subset included under `data/heart+disease/processed.cleveland.data`. The UCI source is [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease). The source file `heart-disease.names` documents its columns and the original 0-to-4 diagnosis. To download the UCI archive when the dataset is absent, run `./scripts/download_dataset.ps1` from PowerShell.

## Run the workflow

Run the EDA and modeling steps:

```powershell
python -m src.eda
python -m src.train
```

Generated plots and summaries are in `artifacts/`. The selected preprocessing-plus-classifier pipeline and its metadata are in `models/`. Training logs the two model runs, metrics, plots, and joblib artifacts to the local MLflow database `mlflow.db`.

Open the MLflow UI in a second terminal:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then visit `http://127.0.0.1:5000`.

Run checks:

```powershell
python -m ruff check src tests
python -m pytest -q
```

## Run the API

The saved model must exist first. Start FastAPI from the project folder:

```powershell
uvicorn src.api:app --reload
```

- API documentation: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Metrics: `http://127.0.0.1:8000/metrics`

The `/predict` body accepts the 13 feature names from the CSV. `ca` and `thal` may be `null`; the fitted preprocessing pipeline imputes them. Do not send real patient information to this learning demo.

To send the included example from PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -InFile sample_request.json
```

## Container and monitoring

Build and run the API image (Docker Desktop's Linux engine must be running):

```powershell
docker build -t heart-disease-api:local .
docker run --rm -p 8000:8000 heart-disease-api:local
```

Or run the API, Prometheus, and provisioned Grafana dashboard together:

```powershell
docker compose up --build
```

Local pages: API `http://127.0.0.1:8000/docs`, Prometheus `http://127.0.0.1:9090`, Grafana `http://127.0.0.1:3000` (local demo credentials: `admin` / `localadmin`). Keep these services local and change the demo password before exposing Grafana to any network. Stop services with Ctrl+C or `docker compose down`.

## Local CI and Kubernetes

- `.github/workflows/ci.yml` at the Git repository root defines Linux GitHub Actions stages for Ruff, pytest, EDA, training, Docker build, and endpoint smoke tests. It uploads training/test artifacts and accepts flat or nested application layouts. Publish it and verify an actual Actions run before claiming CI success.
- `Jenkinsfile` is an alternative Windows Jenkins pipeline. GitHub Actions satisfies the assignment's CI option without requiring Jenkins on the lab VM.
- `scripts/run_local_ci.ps1` runs those checks directly from an activated project environment.
- `deployment/deployment.yaml` defines the Kubernetes Deployment and LoadBalancer Service. Build the Docker image first and ensure the local cluster can access it. Apply only this manifest with `kubectl apply -f deployment/deployment.yaml`; the `deployment/` folder also contains Prometheus/Grafana configuration files.

## Observed model results

On the current fixed stratified 80/20 split (random seed 42), Logistic Regression was selected by mean five-fold training ROC-AUC. Its held-out metrics were accuracy 0.885, precision 0.839, recall 0.929, and ROC-AUC 0.965. Random Forest held-out metrics were 0.869, 0.813, 0.929, and 0.951 respectively. Results are specific to this small historical dataset and split; they should not be interpreted as clinical performance.

## Remaining evidence to collect

The project files are prepared, but final infrastructure proof still depends on running the Docker engine and local Kubernetes/Jenkins services. Capture actual container, monitoring, CI, and deployment screenshots, then write the final report/video using your own walkthrough. Do not state that a service or pipeline ran until you have verified it on your machine.
