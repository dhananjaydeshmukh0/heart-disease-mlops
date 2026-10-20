# Heart Disease Classification — End-to-End MLOps

An end-to-end Machine Learning Operations (MLOps) project that trains, evaluates, tracks, deploys, and monitors a binary heart disease classification model using the UCI Heart Disease dataset.

The project demonstrates exploratory data analysis, preprocessing pipelines, model comparison, MLflow experiment tracking, automated testing, GitHub Actions CI, Docker containerization, Kubernetes deployment, FastAPI inference, and Prometheus-compatible metrics.

> **Disclaimer:** This project is for educational purposes only. Predictions are not medical diagnoses and must not be used to make healthcare decisions.

## Table of Contents

- [Project Objectives](#project-objectives)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Evaluation](#model-evaluation)
- [Getting Started](#getting-started)
- [Run Tests and Linting](#run-tests-and-linting)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Run the Prediction API](#run-the-prediction-api)
- [Docker](#docker)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Monitoring](#monitoring)
- [Continuous Integration](#continuous-integration)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Author and Repository](#author-and-repository)

## Project Objectives

1. Explore and prepare the heart disease dataset.
2. Build a reusable preprocessing pipeline.
3. Train and compare two classification algorithms.
4. Evaluate models using stratified five-fold cross-validation.
5. Track model parameters, metrics, and artifacts using MLflow.
6. Automate code-quality checks and tests using GitHub Actions.
7. Package the inference service with Docker.
8. Deploy the API on a local Kubernetes cluster.
9. Expose health checks, prediction endpoints, and operational metrics.

## Architecture

The project follows this workflow:

```text
UCI Heart Disease Dataset
          |
          v
Exploratory Data Analysis
          |
          v
Data Preprocessing Pipeline
          |
          v
Model Training
  |                   |
  v                   v
Logistic Regression   Random Forest
  |                   |
  +---------+---------+
            |
            v
Cross-Validation and Model Comparison
            |
            v
       MLflow Tracking
            |
            v
      Trained Model
            |
            v
       FastAPI Service
            |
            v
      Docker Container
            |
            v
  Kubernetes Deployment
            |
            v
Prediction API and Monitoring
```

GitHub Actions automates linting, testing, model training, and Docker image building as defined in the CI workflow.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python 3.11 |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Dataset retrieval | ucimlrepo |
| Experiment tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Testing | pytest |
| Code quality | Ruff |
| Containerization | Docker |
| Orchestration | Kubernetes, Docker Desktop |
| CI | GitHub Actions |
| Monitoring | prometheus-fastapi-instrumentator |

## Project Structure

```text
heart-disease-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── raw/
├── deployment/
│   └── kubernetes/
│       └── deployment.yaml
├── models/
├── screenshots/
│   ├── eda/
│   └── mlflow/
├── src/
│   ├── app.py
│   ├── eda.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── Dockerfile
├── requirements.txt
└── README.md
```

Some generated files, folders, or test filenames may differ depending on the final repository contents.

## Dataset

The project uses the UCI Heart Disease dataset, available through the UCI Machine Learning Repository.

Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease

The prepared dataset used in this project contains:

- 303 records
- 13 input features
- One binary target variable
- 164 records in target class 0
- 139 records in target class 1
- Missing values in `ca` and `thal`, handled during preprocessing

The target mapping should be interpreted according to the mapping implemented in the project's data preparation code.

## Machine Learning Pipeline

### Exploratory Data Analysis

The EDA workflow examines:

- Target class distribution
- Feature distributions
- Correlations between numerical and encoded features
- Missing values and dataset characteristics

Generated plots are saved for analysis and experiment tracking.

### Preprocessing

**Numerical features:**

- `age`
- `trestbps`
- `chol`
- `thalach`
- `oldpeak`

Numerical preprocessing uses median imputation and standardization.

**Categorical features:**

- `sex`
- `cp`
- `fbs`
- `restecg`
- `exang`
- `slope`
- `ca`
- `thal`

Categorical preprocessing uses most-frequent imputation and one-hot encoding with unknown categories handled safely.

### Models

Two classifiers are compared:

1. Logistic Regression
2. Random Forest Classifier

Model evaluation uses stratified five-fold cross-validation with accuracy, precision, recall, and ROC-AUC.

## Model Evaluation

The recorded mean cross-validation results are:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.8547 | 0.8383 |
| Precision | 0.8744 | 0.8589 |
| Recall | 0.7979 | 0.7767 |
| ROC-AUC | 0.9188 | 0.9176 |

Logistic Regression achieved higher mean values for all four reported metrics in this evaluation. These are cross-validation results on the project dataset, not a guarantee of performance on new or external data.

## Getting Started

### Prerequisites

Install the following:

- Python 3.11
- Git
- Docker Desktop
- Kubernetes enabled in Docker Desktop for Kubernetes deployment

### 1. Clone the repository

```powershell
git clone https://github.com/dhananjaydeshmukh0/heart-disease-mlops.git
cd heart-disease-mlops
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the Python executable directly from the virtual environment or follow your system's virtual-environment policy.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run exploratory data analysis

```powershell
python -m src.eda
```

Check `src/eda.py` for any required dataset download or output-directory configuration.

### 5. Train and evaluate models

```powershell
python -m src.train
```

The training workflow evaluates both classifiers, records results using MLflow, and saves model and evaluation artifacts according to the project configuration.

## Run Tests and Linting

Run the automated tests:

```powershell
python -m pytest tests/ -v
```

Run Ruff:

```powershell
ruff check src tests
```

Both commands should complete successfully before submitting changes.

## MLflow Experiment Tracking

The MLflow experiment is named:

`Heart-Disease-Classification`

The training script is configured to record model runs, parameters, cross-validation metrics, and available model and evaluation artifacts.

After running training, start the MLflow UI from the project directory:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open:

http://localhost:5000

Select the `Heart-Disease-Classification` experiment to inspect recorded runs.

**Note:** The tracking database must exist at the configured location. If your code uses a different tracking URI, use that configuration instead.

## Run the Prediction API

Start the API locally:

```powershell
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

### Available endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Check API health |
| `/predict` | POST | Generate a model prediction |
| `/metrics` | GET | Expose operational metrics |

The exact response schema is defined in `src/app.py`.

### Example prediction request

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 120,
  "exang": 1,
  "oldpeak": 3.0,
  "slope": 0,
  "ca": 2,
  "thal": 3
}
```

Submit the request through Swagger UI to see the actual prediction returned by the model. The example is illustrative and does not represent a real patient assessment.

## Docker

Build the Docker image:

```powershell
docker build -t heart-disease-api:local .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 heart-disease-api:local
```

Open http://localhost:8000/docs to test the API.

The Dockerfile copies the application source and model artifacts into the image. Ensure the expected model file exists before building.

If your Kubernetes manifest references a Docker Hub image, make sure the image name and tag in the manifest match the image you have built and pushed.

## Kubernetes Deployment

Prerequisites:

- Docker Desktop running
- Kubernetes enabled
- `kubectl` configured to use the Docker Desktop cluster
- A container image accessible to the cluster

Apply the deployment manifest:

```powershell
kubectl apply -f deployment/kubernetes/deployment.yaml
```

Check deployment status:

```powershell
kubectl get deployments
kubectl get pods
kubectl get services
kubectl rollout status deployment/heart-disease-api
```

Forward the service port to your local machine:

```powershell
kubectl port-forward service/heart-disease-api 8001:8000
```

While port-forwarding is running, open:

- Swagger UI: http://127.0.0.1:8001/docs
- Health endpoint: http://127.0.0.1:8001/health
- Metrics: http://127.0.0.1:8001/metrics

The port-forward must remain active while you use these addresses. They are local access addresses, not public deployment URLs.

## Monitoring

The FastAPI application uses `prometheus-fastapi-instrumentator` to expose Prometheus-compatible service metrics.

Open:

http://127.0.0.1:8001/metrics

The output may include HTTP request counters, request-duration measurements, and other runtime metrics depending on the instrumentation configuration.

Send requests to the API before checking the metrics if you want to demonstrate request activity.

## Continuous Integration

The GitHub Actions workflow is located at:

`.github/workflows/ci.yml`

The workflow is configured to perform project quality checks, including:

- Installing Python dependencies
- Running Ruff
- Running pytest
- Executing the training script
- Building a Docker image
- Uploading configured workflow artifacts

Check the **Actions** tab in the GitHub repository to inspect workflow results.

## Limitations and Future Improvements

- The dataset is relatively small and may not represent all populations.
- Cross-validation does not establish clinical validity.
- The deployment is a local academic demonstration.
- Current monitoring focuses on service-level metrics rather than complete model-drift detection.
- Production use would require additional security, privacy, validation, governance, and operational controls.

Possible future improvements include external validation, model calibration, drift detection, stronger deployment automation, and more comprehensive monitoring.

## Author and Repository

**GitHub repository:**  
https://github.com/dhananjaydeshmukh0/heart-disease-mlops

**Author:** [Enter your name]

**Course:** [Enter your official course name and code]

## References

- UCI Heart Disease dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
- MLflow: https://mlflow.org/docs/latest/
- scikit-learn: https://scikit-learn.org/stable/
- FastAPI: https://fastapi.tiangolo.com/
- Docker: https://docs.docker.com/
- Kubernetes: https://kubernetes.io/docs/
- Prometheus: https://prometheus.io/docs/
