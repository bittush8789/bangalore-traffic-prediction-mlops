# 🚦 Bangalore Traffic Prediction MLOps Platform

[![MLOps Pipeline](https://github.com/bittush8789/bangalore-traffic-prediction-mlops/actions/workflows/mlops-pipeline.yml/badge.svg)](https://github.com/bittush8789/bangalore-traffic-prediction-mlops/actions/workflows/mlops-pipeline.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Production-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![ArgoCD](https://img.shields.io/badge/ArgoCD-GitOps-EF7B4D?style=for-the-badge&logo=argo&logoColor=white)](https://argoproj.github.io/argo-cd/)

A production-grade, end-to-end MLOps platform for real-time Bangalore traffic intelligence. This project demonstrates a complete CI/CD/CT (Continuous Training) lifecycle using industry-standard tools like Kubernetes, KServe, ArgoCD, and AWS.

---

## 🏗️ Architecture Flow

1. **Development**: Developer pushes code to GitHub.
2. **CI/CD Pipeline**: GitHub Actions triggers:
   - Environment setup & Dependency installation.
   - **DVC Pipeline**: Synthetic data generation (50k+ rows) & Feature Engineering.
   - **Model Training**: XGBoost/LightGBM training with **MLflow** experiment tracking.
   - **Artifact Storage**: Serialized models (.pkl) uploaded to **AWS S3**.
3. **Containerization**: Docker image built and pushed to **Amazon ECR**.
4. **GitOps Deployment**:
   - Pipeline automatically updates `k8s/inference.yaml` with the latest S3 model URI.
   - **ArgoCD** detects the Git change and synchronizes the state to the Kubernetes cluster.
5. **Serving**: **KServe** deploys the model as a scalable InferenceService.
6. **Monitoring**: **Prometheus** scrapes metrics from the FastAPI app, visualized in **Grafana**.

---

## 🛠️ Tech Stack

- **Inference**: FastAPI, Uvicorn, Pydantic
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, Pandas, NumPy
- **Orchestration**: Kubernetes, KServe, ArgoCD
- **Cloud Infrastructure**: AWS (S3, ECR, EKS)
- **Data/Model Management**: DVC, MLflow, Joblib
- **CI/CD**: GitHub Actions, Docker
- **Observability**: Prometheus, Grafana

---

## 📁 Project Structure

```text
bangalore-traffic-prediction-mlops/
│── .github/workflows/       # CI/CD pipeline definitions
│── app.py                   # FastAPI Production Server
│── train.py                 # Model training & MLflow logic
│── generate_data.py         # Synthetic data engine
│── requirements.txt         # Production dependencies
│── Dockerfile               # Container definition
│── dvc.yaml                 # Data pipeline versioning
│── models/                  # Local model artifacts (ignored by git)
│── data/                    # Datasets (ignored by git)
│── templates/               # Glassmorphism UI (HTML)
│── static/                  # UI Assets (CSS/JS)
│── k8s/                     # Kubernetes Manifests (KServe)
│── argocd/                  # GitOps Application definitions
│── monitoring/              # Prometheus/Grafana configs
│── README.md                # Platform documentation
```

---

## 🚀 Getting Started

### 1. Local Development
```bash
# Clone the repo
git clone https://github.com/bittush8789/bangalore-traffic-prediction-mlops.git
cd bangalore-traffic-prediction-mlops

# Setup environment
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run pipeline
python generate_data.py
python train.py

# Start FastAPI
uvicorn app:app --reload
```

### 2. Docker Execution
```bash
docker build -t traffic-predictor .
docker run -p 8000:8000 traffic-predictor
```

### 3. Local Kubernetes with Kind
For local MLOps testing, you can use **Kind** (Kubernetes in Docker):
```bash
# 1. Install Kind (if not installed)
# choco install kind (Windows) or brew install kind (macOS)

# 2. Create a local cluster
kind create cluster --name blr-traffic-cluster

# 3. Load your local Docker image into Kind
kind load docker-image traffic-predictor:latest --name blr-traffic-cluster

# 4. Apply Kubernetes manifests
kubectl apply -f k8s/serviceaccount.yaml
kubectl apply -f k8s/inference.yaml
```

### 4. Production Kubernetes & ArgoCD Deployment
1. Ensure `kubectl` is connected to your EKS/GKE cluster.
2. Install ArgoCD and apply the application manifest:
   ```bash
   kubectl apply -f argocd/application.yaml
   ```
3. ArgoCD will automatically deploy the **KServe InferenceService** defined in `k8s/inference.yaml`.

---

## 📡 API Endpoints

- **Predict**: `POST /predict` - Real-time traffic inference.
- **Health**: `GET /health` - Liveness/Readiness probe.
- **Metrics**: `GET /metrics` - Prometheus metrics export.
- **Docs**: `GET /docs` - Interactive Swagger documentation.

---

## 🧪 Experiment Tracking (MLflow)
Experiments are logged locally (or to a remote tracking server). You can visualize metrics by running:
```bash
mlflow ui
```

---

## 🛡️ Security & Scalability
- **Secrets**: AWS credentials and model paths are handled via GitHub Secrets and K8s Secrets.
- **Scalability**: KServe enables auto-scaling based on request volume (Serverless inference).
- **Versioning**: DVC ensures data and model lineage are maintained.

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or pull request for any improvements.

---
**Developed by [Your Name] | Dedicated to Scalable AI Solutions**
