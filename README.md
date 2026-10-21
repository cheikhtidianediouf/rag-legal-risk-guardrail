# ⚖️ RAG Legal & Financial Risk Guardrail

Une application MLOps et LLM prête pour la production, conçue pour analyser des contrats juridiques complexes, les indexer dans une base vectorielle locale, effectuer une recherche sémantique (RAG) et auditer automatiquement les accords pour détecter les risques financiers et juridiques à l'aide d'un garde-fou personnalisé (**Risk Guardrail**).

---

## 🚀 Fonctionnalités Clés

* **Pipeline de Données Automatisé :** Télécharge et traite les contrats juridiques depuis le jeu de données officiel CUAD (*Contract Understanding Atticus Dataset*) via Hugging Face.
* **Parsing PDF Avancé :** Extrait le texte intégral de documents juridiques multi-pages grâce à la bibliothèque `pdfplumber`.
* **Recherche Sémantique (RAG) :** Découpe les contrats longs en morceaux (*chunks*) et les indexe à l'aide de `Sentence-Transformers` (`all-MiniLM-L6-v2`) au sein d'une base vectorielle persistante `ChromaDB`.
* **Garde-fou de Risques (*Risk Guardrail*) :** Scanne automatiquement les clauses récupérées à la recherche de mots-clés déclencheurs de risques juridiques et financiers (ex: *unlimited liability*, *damages*, *penalties*) pour classifier le niveau de risque (**Élevé**, **Moyen**, **Faible**) avec des recommandations exploitables.
* **Interface Web Interactive :** Développée avec **Streamlit** pour une expérience utilisateur fluide et en temps réel.

---

## 🛠️ Stack Technique

* **Langage :** Python 3.12
* **Traitement des données :** Pandas, Hugging Face `datasets`, `pdfplumber`
* **Embeddings & Base Vectorielle :** `sentence-transformers`, `chromadb`
* **Interface & Ops :** `streamlit`, Architecture projet modulaire MLOps

---

## 📂 Structure du Projet

```text
Proet RAG-RISK/
│
├── data/
│   ├── raw/                 # Fichiers bruts du dataset
│   ├── processed/           # CSVs nettoyés (cuad_processed.csv)
│   └── vector_db/           # Stockage persistant ChromaDB
│
├── src/
│   ├── data_loader.py       # Pipeline d'ingestion et de parsing PDF
│   ├── indexer.py           # Chunking et génération des embeddings vectoriels
│   └── rag_pipeline.py      # Moteur de recherche sémantique + Risk Guardrail
│
├── venv/                    # Environnement virtuel Python
├── app.py                   # Application Web Streamlit
└── README.md                # Documentation du projet


⚙️ Installation & Configuration
Cloner le dépôt :

git clone [https://github.com/votre-nom-dutilisateur/rag-legal-risk-guardrail.git](https://github.com/votre-nom-dutilisateur/rag-legal-risk-guardrail.git)
cd Proet RAG-RISK

Créer et activer un environnement virtuel (Python 3.12 recommandé) :

python -m venv venv
venv\Scripts\activate  # Sous Windows

Installer les dépendances :

pip install pandas datasets pdfplumber sentence-transformers chromadb streamlit

🚀 Guide d'Exécution : 

Étape 1 : Ingestion et Traitement des Contrats
Télécharge et parse les contrats PDF bruts pour les structurer :

python src/data_loader.py

Étape 2 : Construction de l'Index Vectoriel
Découpe le texte et stocke les embeddings en local dans ChromaDB :

python src/indexer.py

Étape 3 : Tester le Pipeline RAG & Guardrail en CLI
Exécute une requête de recherche sémantique et une évaluation des risques en console :

python src/rag_pipeline.py

Étape 4 : Lancer l'Interface Web (Streamlit)
Démarre le tableau de bord interactif :

streamlit run app.py


