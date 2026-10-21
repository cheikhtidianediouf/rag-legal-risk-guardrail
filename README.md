RAG Legal & Financial Risk Guardrail
Système MLOps complet d'analyse de conformité contractuelle avec RAG, évaluation de qualité en temps réel et détection d'hallucinations.

📌 Context & Problem Statement
Dans les secteurs juridique et financier, l'utilisation de modèles d'IA générative (LLM) pour analyser des contrats complexes ou des rapports financiers présente un risque critique : les hallucinations et l'imprécision des réponses. Une clause mal interprétée ou une fausse information générée peut entraîner des conséquences juridiques et financières majeures.

RAG Legal Risk Guardrail résout ce problème en combinant une architecture RAG (Retrieval-Augmented Generation), un modèle de scoring prédictif (XGBoost) et un pipeline de suivi MLOps (MLflow, Docker, CI/CD). Le système évalue en temps réel la fiabilité des réponses générées avant leur présentation à l'utilisateur final.

🎯 Key Features
Retrieval-Augmented Generation (RAG) : Extraction automatique et ciblée de clauses juridiques à partir du benchmark CUAD (Contract Understanding Atticus Dataset).

Dual-Layer Guardrail System :

Évaluation dynamique LLM-as-a-Judge : Calcul des scores de pertinence (context recall, faithfulness, answer relevancy).

Predictive Quality Scoring (XGBoost) : Modèle Machine Learning supervisé prédisant le niveau de risque ou d'hallucination d'une réponse à partir de métriques comportementales et statistiques.

MLOps & Observability : Tracking complet des expériences, des métriques et du versionnement de modèles via MLflow.

Production-Ready Architecture :

API RESTful (FastAPI) exposant les fonctionnalités d'analyse et d'évaluation.

Dashboard Interactif (Streamlit) pour la démonstration temps réel.

Containerisation (Docker) & Intégration Continue (GitHub Actions).

🛠️ Tech Stack
Language & Core : Python 3.10+, Pandas, NumPy, Pydantic

RAG & Vector Storage : LangChain / LlamaIndex, ChromaDB, Sentence-Transformers

ML & Evaluation : XGBoost, Scikit-Learn, LLM-as-a-Judge

MLOps & Monitoring : MLflow, Pytest

API & Frontend : FastAPI, Uvicorn, Streamlit

DevOps : Docker, GitHub Actions


🏗️ System Architecture

				  +---------------------------------------+
                                  |         Contract / Document           |
                                  +---------------------------------------+
                                                      |
                                                      v
                                      +-------------------------------+
                                      |    VectorDB & RAG Engine      |
                                      +-------------------------------+
                                                      |
                                                      v
                                      +-------------------------------+
                                      |     Generated LLM Answer      |
                                      +-------------------------------+
                                                      |
                                                      v
+---------------------------------+   +-------------------------------+
| Metrics Extraction (RAGAS/Lang) |-->|   XGBoost Quality Guardrail   |
+---------------------------------+   +-------------------------------+
                                                      |
                                                      v
                                      +-------------------------------+
                                      | Score de Risque & Validation  |
                                      +-------------------------------+
                                                      |
                                                      v
                                      +-------------------------------+
                                      |   FastAPI & Streamlit UI      |
                                      +-------------------------------+


📁 Repository Structure

rag-legal-risk-guardrail/
├── data/                  # Jeux de données (Raw & Processed CUAD)
├── notebooks/             # Exploratoire, prototypage RAG & XGBoost
├── src/
│   ├── data_loader.py     # Ingestion & préparation des données
│   ├── rag_engine.py      # Pipeline RAG & Vector Database (ChromaDB)
│   ├── evaluator.py       # Extraction de métriques & Modèle XGBoost
│   └── api.py             # API REST (FastAPI)
├── app/
│   └── dashboard.py       # Interface utilisateur (Streamlit)
├── tests/                 # Tests unitaires & d'intégration (Pytest)
├── Dockerfile             # Containerisation pour la mise en production
├── requirements.txt       # Dépendances du projet
└── README.md              # Documentation du projet