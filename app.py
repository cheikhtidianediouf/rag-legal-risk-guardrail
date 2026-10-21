import os
import streamlit as st
from src.rag_pipeline import LegalRiskRAGPipeline

# Configuration de la page Streamlit
st.set_page_config(
    page_title="RAG Legal & Financial Risk Guardrail",
    page_icon="⚖️",
    layout="wide",
)

# Initialisation du pipeline RAG avec mise en cache pour optimiser les performances
@st.cache_resource
def load_pipeline():
    return LegalRiskRAGPipeline()

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du pipeline : {e}")
    st.stop()

# En-tête de l'application
st.title("⚖️ RAG Legal & Financial Risk Guardrail")
st.markdown("Plateforme MLOps d'analyse juridique et de détection automatique des risques contractuels.")

# Barre latérale (Sidebar)
st.sidebar.header("⚙️ Configuration & Infos")
st.sidebar.info(
    "**Stack Technique :**\n"
    "- Modèle : `all-MiniLM-L6-v2`\n"
    "- Base vectorielle : `ChromaDB`\n"
    "- Jeu de données : `CUAD (The Atticus Project)`\n"
    "- Sécurité : `Risk Guardrail actif`"
)

# Zone de saisie utilisateur
query = st.text_input(
    "Posez votre question juridique (ex: *What are the liabilities and damages clauses?*):",
    value="What are the liabilities and damages clauses in this contract?"
)

n_results = st.slider("Nombre d'extraits de contrats à analyser :", min_value=1, max_value=5, value=2)

# Bouton d'action
if st.button("🚀 Lancer l'analyse des risques", type="primary"):
    if not query.strip():
        st.warning("Veuillez entrer une question valide.")
    else:
        with st.spinner("Recherche sémantique dans ChromaDB et analyse par le Garde-fou en cours..."):
            chunks = pipeline.search_relevant_clauses(query, n_results=n_results)

        if not chunks:
            st.warning("Aucun extrait pertinent trouvé dans la base vectorielle.")
        else:
            st.success(f"🔍 Analyse terminée : {len(chunks)} extraits de contrats identifiés.")
            
            for i, item in enumerate(chunks, 1):
                # Utilisation d'expanders pour structurer l'affichage propre de chaque contrat
                with st.expander(f"📄 Résultat #{i} | Contrat source : `{item['contract_id']}`", expanded=True):
                    
                    col1, col2 = st.columns([1.5, 1])
                    
                    with col1:
                        st.markdown("**Extrait brut du contrat :**")
                        st.text_area(
                            f"Texte_{i}", 
                            value=item["text"], 
                            height=180, 
                            disabled=True, 
                            label_visibility="collapsed"
                        )
                    
                    with col2:
                        # Évaluation par le Risk Guardrail
                        risk_eval = pipeline.evaluate_legal_risk(item["text"])
                        
                        st.markdown("**🛡️ Rapport du Risk Guardrail :**")
                        
                        # Affichage visuel dynamique selon le niveau de risque
                        risk_level = risk_eval["risk_level"]
                        if "ÉLEVÉ" in risk_level:
                            st.error(f"**Niveau :** {risk_level}")
                        elif "MOYEN" in risk_level:
                            st.warning(f"**Niveau :** {risk_level}")
                        else:
                            st.success(f"**Niveau :** {risk_level}")
                            
                        st.markdown(f"**Indicateurs détectés :** `{risk_eval['triggers'] if risk_eval['triggers'] else 'Aucun'}`")
                        st.markdown(f"**Recommandation :**\n> {risk_eval['recommendation']}")