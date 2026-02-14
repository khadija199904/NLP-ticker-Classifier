#  NLP Ticket Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-FA5252?logo=chromadb&logoColor=white)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Visualization-F46800?logo=grafana&logoColor=white)](https://grafana.com/)
[![Evidently AI](https://img.shields.io/badge/Evidently%20AI-ML%20Monitoring-4B0082)](https://www.evidentlyai.com/)

A complete NLP pipeline for support ticket classification using **Hugging Face**, **ChromaDB**, and orchestrated with **MLOps** monitoring tools.

---

##  Table of Contents
- [Technologies](#-technologies)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Monitoring](#-monitoring)

---

##  Technologies
*   **Core**: Python
*   **Data Analysis**: Pandas, Matplotlib, Seaborn
*   **NLP & ML**: NLTK, Scikit-learn, LangChain, Hugging Face Transformers
*   **Database**: ChromaDB (Vector Store)
*   **Containerization**: Docker, Docker Compose
*   **Monitoring**: Evidently AI, Prometheus, Grafana, cAdvisor, Node Exporter

---

##  Project Structure
```text
NLP-ticker-Classifier/
├── .github/workflows/      # CI/CD Pipelines
├── config.py               # Application Configuration
├── data/                   # Data Storage
│   ├── raw/                # Original datasets
│   └── processed/          # Cleaned & processed data
├── docker-compose.yml      # Docker services definition
├── infrastructure/         # Infrastructure configurations
│   ├── k8s/                # Kubernetes manifests (Deployments/Services)
│   └── monitoring/         # Grafana & Prometheus configs
├── notebooks/              # Jupyter Notebooks for EDA & Prototyping
├── requirements.txt        # Project Dependencies
├── src/                    # Source Code
│   ├── preprocessing.py    # Text cleaning & preparation
│   ├── vectorization.py    # Embedding with HuggingFace
│   ├── train_model.py      # Model training logic
│   ├── predict.py          # Inference script
│   └── monitoring.py       # Data drift monitoring
└── README.md               # Project Documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional, for full stack)

### 1. Local Installation (Python)
```bash
# Clone the repository
git clone <repository-url>
cd NLP-ticker-Classifier

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run with Docker (Recommended)
To start the entire application including the monitoring stack:
```bash
docker-compose up --build -d
```

---

## 💻 Usage

### Training the Model
Run the training script to process data and train the classifier:
```bash
python src/train_model.py
```

### Making Predictions
Use the prediction script to classify new tickets:
```bash
python src/predict_model.py
```

---

## 📊 Monitoring

The project includes a full monitoring stack accessible via localhost when running with Docker:

| Service | URL | Description |
|---------|-----|-------------|
| **Grafana** | [http://localhost:3000](http://localhost:3000) | Visual Dashboards |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | Metric Collection |
| **cAdvisor** | [http://localhost:8080](http://localhost:8080) | Container Metrics |
| **ChromaDB** | [http://localhost:8000](http://localhost:8000) | Vector Database API |

---

## 📜 License
[MIT License](LICENSE)
