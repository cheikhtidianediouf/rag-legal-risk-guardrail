import os
import chromadb
from sentence_transformers import SentenceTransformer


class LegalRiskRAGPipeline:
    """Pipeline RAG combinant la recherche sémantique dans ChromaDB

    et un système de Garde-fou (Risk Guardrail) pour l'analyse juridique.
    """

    def __init__(self, db_dir: str = "data/vector_db"):
        self.db_dir = db_dir
        
        print("🔄 Chargement du modèle d'embedding pour la recherche...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("🗄️ Connexion à la base vectorielle ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.chroma_client.get_or_create_collection(name="cuad_contracts")

    def search_relevant_clauses(self, query: str, n_results: int = 3) -> list[dict]:
        """Recherche les chunks les plus proches sémantiquement de la requête."""
        query_embedding = self.model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        retrieved_chunks = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            
            for doc, meta in zip(docs, metadatas):
                retrieved_chunks.append({
                    "contract_id": meta.get("contract_id", "Inconnu"),
                    "text": doc
                })
                
        return retrieved_chunks

    def evaluate_legal_risk(self, chunk_text: str) -> dict:
        """Garde-fou (Risk Guardrail) : Analyse le texte pour détecter des risques juridiques/financiers."""
        text_lower = chunk_text.lower()
        
        # Mots-clés déclencheurs de risques juridiques et financiers
        high_risk_keywords = ["unlimited liability", "indemnify", "damages", "penalty", "exclusive", "termination without cause"]
        medium_risk_keywords = ["governing law", "jurisdiction", "confidentiality", "warranty", "expire"]
        
        risk_score = 0
        detected_triggers = []

        for kw in high_risk_keywords:
            if kw in text_lower:
                risk_score += 2
                detected_triggers.append(kw)
                
        for kw in medium_risk_keywords:
            if kw in text_lower:
                risk_score += 1
                detected_triggers.append(kw)

        # Classification du niveau de risque
        if risk_score >= 4:
            risk_level = "🚨 ÉLEVÉ (High Risk)"
            recommendation = "Attention : Clause restrictive ou responsabilité potentiellement lourde. Relecture par un juriste conseillée."
        elif risk_score >= 2:
            risk_level = "⚠️ MOYEN (Medium Risk)"
            recommendation = "Vérifier les conditions de mise en œuvre et les obligations associées."
        else:
            risk_level = "✅ FAIBLE (Low Risk)"
            recommendation = "Clause standard ou neutre."

        return {
            "risk_level": risk_level,
            "triggers": list(set(detected_triggers)),
            "recommendation": recommendation
        }

    def run_query(self, query: str):
        """Exécute le pipeline complet RAG + Guardrail pour une question donnée."""
        print(f"\n🔍 Question posée : \"{query}\"")
        print("=" * 60)
        
        chunks = self.search_relevant_clauses(query, n_results=2)
        
        if not chunks:
            print("❌ Aucun extrait pertinent trouvé dans la base.")
            return

        for i, item in enumerate(chunks, 1):
            print(f"\n📄 Résultat #{i} | Contrat source : {item['contract_id']}")
            print("-" * 40)
            print(f"Extrait du contrat :\n{item['text'][:400]}...\n")
            
            # Analyse par le Risk Guardrail
            risk_eval = self.evaluate_legal_risk(item['text'])
            print(f"🛡️ Analyse du Garde-fou (Risk Guardrail) :")
            print(f"  • Niveau de risque : {risk_eval['risk_level']}")
            print(f"  • Indicateurs détectés : {risk_eval['triggers'] if risk_eval['triggers'] else 'Aucun'}")
            print(f"  • Recommandation : {risk_eval['recommendation']}")
            print("=" * 60)


if __name__ == "__main__":
    pipeline = LegalRiskRAGPipeline()
    
    # Test avec une question juridique type
    pipeline.run_query("What are the liabilities and damages clauses in this contract?")