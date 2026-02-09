# 🎫 NLP Ticket Classifier

A complete NLP pipeline for support ticket classification using **Hugging Face**, **ChromaDB**, and orchestrated with **Kubernetes** and **MLOps** monitoring tools (**Evidently AI**, **Prometheus**, **Grafana**).

---

## 🏗️ Project Structure

```text
NLP-ticker-Classifier/
├── .github/workflows/      # CI/CD: Linting & Docker Build
├── config/                 # Configuration files (YAML/JSON)
├── data/                   
│   ├── raw/                # Original dataset (read-only)
│   └── processed/          # Cleaned NLP data
├── k8s/                    # Kubernetes Manifests (Jobs/CronJobs)
├── ml/                     # Research Notebooks (EDA, Prototyping)
├── monitoring/             
│   ├── grafana/            # Dashboards configurations
│   └── prometheus/         # Metrics scraping config
├── src/                    # Production-ready Source Code
│   ├── preprocessing.py    # NLP text cleaning
│   ├── vectorization.py    # Embeddings & ChromaDB indexing
│   ├── train.py            # Model training & optimization
│   └── monitoring.py       # Drift analysis (Evidently AI)
├── Dockerfile              # Container definition
├── docker-compose.yml      # Local monitoring stack (Grafana/Prometheus)
├── requirements.txt        # Python dependencies
└── jira_plan.md            # Detailed project road-map
```

---

## 🛠️ Technology Stack

*   **NLP Core**: Hugging Face (Embeddings), Scikit-Learn (Classification).
*   **Storage**: ChromaDB (Vector Database).
*   **MLOps**: Evidently AI (Data/Prediction Drift).
*   **Infrastructure**: Docker, Kubernetes (Minikube).
*   **Supervision**: Prometheus, Grafana, cAdvisor, Node Exporter.

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Local Infrastructure (Monitoring)
To launch the supervision stack (Grafana, Prometheus):
```bash
docker-compose up -d
```
*   **Grafana**: [http://localhost:3000](http://localhost:3000)
*   **Prometheus**: [http://localhost:9090](http://localhost:9090)

---

## 📅 Planning
The project is organized into **5 Epics** tracked in the `jira_plan.md` file.
*   **Duration**: 09/02/2026 - 14/02/2026.
*   **Goal**: Industrialization of a Batch NLP Pipeline.