#  NLP Ticket Classifier : Industrialisation MLOps

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-FA5252?logo=chromadb&logoColor=white)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Visualization-F46800?logo=grafana&logoColor=white)](https://grafana.com/)
[![Evidently AI](https://img.shields.io/badge/Evidently%20AI-ML%20Monitoring-4B0082)](https://www.evidentlyai.com/)

Ce projet propose une solution industrielle pour la classification automatique de tickets de support IT. À partir d'un historique de **62 609 emails**, le pipeline transforme le texte en représentations sémantiques pour prédire le type de ticket (*Incident, Request, Problem, Change*). 

L'accent est mis sur la **robustesse du pipeline** (Docker, Kubernetes) et la **surveillance continue** (Evidently AI, Prometheus, Grafana).

---


##  Table of Contents
- [Technologies](#technologies)
- [Fonctionnalités du Pipeline](#fonctionnalités-du-pipeline)
- [Structure du Projet](#structure-du-projet)
- [Installation et Lancement](#installation-et-lancement)
    - [1. Pré-requis](#1-pré-requis)
    - [2. Démarrage de la Stack](#2-démarrage-de-la-stack)
    - [3. Exécution du Pipeline](#3-exécution-du-pipeline-orchestrateur)
- [Monitoring & Observabilité](#monitoring--observabilité)
    - [Qualité du Modèle (ML Metrics)](#qualité-du-modèle-ml-metrics)
    - [Santé de l'Infrastructure](#santé-de-linfrastructure)
- [Déploiement Kubernetes (Minikube)](#déploiement-kubernetes-minikube)
- [Auteur](#auteur)

## Technologies
*   **Core**: Python
*   **Data Analysis**: Pandas, Matplotlib, Seaborn
*   **NLP** : Hugging Face (`paraphrase-multilingual-MiniLM-L12-v2`), NLTK.
*   **Machine Learning** : Scikit-Learn (Logistic Regression ).
*   **Base de données** : ChromaDB (Vector Database).
*   **Monitoring ML** : Evidently AI (Data Drift & Classification Performance).
*   **Monitoring Infra** : Prometheus, Grafana, cAdvisor, Node Exporter.
*   **Conteneurisation** : Docker, Docker Compose.


##  Fonctionnalités du Pipeline
1.  **NLP Prep**: Fusion `subject` + `body`, nettoyage (stopwords, ponctuation, normalisation).
2.  **Semantic Embedding**: Transformation du texte en vecteurs via le modèle Hugging Face `paraphrase-multilingual-MiniLM-L12-v2`.
3.  **Vector Storage**: Indexation dans **ChromaDB** avec gestion par lots (batching) pour garantir la montée en charge (62k+ documents).
4.  **Multi-Model Training**: Comparaison automatique entre **Logistic Regression** et **SVM (LinearSVC)** avec sauvegarde du meilleur modèle (*Best Model Registry*).
5.  **ML Monitoring**: Analyse du **Data Drift** et des performances de classification avec **Evidently AI**.
6.  **Infra Monitoring**: Surveillance en temps réel du CPU/RAM via **Prometheus**, **Grafana**, **cAdvisor** et **Node Exporter**.

---


##  Structure du Projet
```text

NLP-ticker-Classifier/
├── src/                        # Code source du pipeline
│   ├── embedding_service.py    # Vectorisation & Ingestion ChromaDB
│   ├── train_eval.py           # Entraînement et comparaison de modèles
│   └── monitoring.py           # Génération des rapports Evidently AI
├── data/                       # Volumes de données
│   └── data_prepared.csv       # Dataset nettoyé
├── infrastructure/             # Configurations DevOps
│   ├── k8s/                    # Manifests Kubernetes
│   └── monitoring/             # Config Prometheus & Grafana
├── artifacts/                  # Sorties du pipeline
│   ├── models/                 # Modèles entraînés (.joblib)
│   └── reports/                # Rapports de monitoring (HTML)
├── notebooks/                  # Analyse exploratoire (EDA)
├── config.py                   # Configuration centralisée (Paths, Hosts)
├── .env.example
├── requierements.txt
├── Dockerfile                  # Image du pipeline
└── docker-compose.yml          # Orchestration de la stack complète
```

---

##  Installation et Lancement

### 1. Pré-requis
 Python 3.10+
- Docker & Docker Compose
- Un token Hugging Face (à placer dans un fichier `.env`)

###  Installation (Python)
```bash
# Clone the repository
git clone https://github.com/khadija199904/NLP-ticker-Classifier.git
cd NLP-ticker-Classifier

### 2. Démarrage de la Stack
```bash
docker compose up -d --build
```
*Le conteneur `nlp_app` est configuré en mode passif (`tail -f /dev/null`) pour vous laisser piloter le pipeline manuellement.*

### 3. Exécution du Pipeline (Orchestrateur)

```bash
docker compose exec nlp_app python /app/src/main.py
```

---

## Monitoring & Observabilité

### Qualité du Modèle (ML Metrics)
Les rapports de performance montrent une **Accuracy de ~77%**. Le rapport interactif est disponible après exécution du monitoring :
```bash
# Pour visualiser le rapport (serveur local sur port 8001)
google-chrome ~/Desktop/Your-Project/NLP-ticker-Classifier/artifacts/reports/monitoring_report.html
```

### Santé de l'Infrastructure
| Outil | URL | Utilisation |
|-------|-----|-------------|
| **Grafana** | `http://localhost:3000` | Dashboards **14282** (Docker) & **1860** (Host) |
| **Prometheus**| `http://localhost:9090` | Consultation des métriques brutes |
| **cAdvisor**  | `http://localhost:8085` | Stats temps réel des containers |

---

##  Déploiement Kubernetes (Minikube)
Le projet supporte le déploiement sous forme de **Job Kubernetes** pour les exécutions batch en production.
1.  Appliquer le stockage : `kubectl apply -f infrastructure/k8s/storage.yaml`
2.  Lancer ChromaDB : `kubectl apply -f infrastructure/k8s/chromadb-deployment.yaml`
3.  Exécuter le pipeline : `kubectl apply -f infrastructure/k8s/nlp-job.yaml`

---

##  Auteur
**Khadija Elabbioui** - Data Scientist / AI Developer / ML Engineer



