import os
import pandas as pd
from datasets import load_dataset


class CUADDataLoader:
    """Charge, parse les PDF et prépare le jeu de données CUAD

    depuis le dépôt officiel Hugging Face 'theatticusproject/cuad'.
    """

    def __init__(self, processed_data_dir: str = "data/processed"):
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def load_raw_dataset(self, split: str = "train"):
        """Charge le dataset CUAD standard depuis Hugging Face."""
        print(f"🔄 Chargement du dataset CUAD ('theatticusproject/cuad', split='{split}')...")
        dataset = load_dataset(
            "theatticusproject/cuad", 
            split=split, 
            verification_mode="no_checks"
        )
        print(f"✅ {len(dataset)} éléments chargés avec succès.")
        return dataset

    def process_and_extract_clauses(self, limit: int = 10) -> pd.DataFrame:
        """Extrait le texte réel des PDF et structure les données pour le RAG."""
        dataset = self.load_raw_dataset()
        extracted_records = []

        print(f"⚙️ Traitement et extraction du texte des {limit} premiers contrats...")
        for i, item in enumerate(dataset):
            if i >= limit:
                break

            pdf_obj = item.get("pdf")
            contract_id = f"Contract_{i}"
            full_text = ""

            # Extraction du texte via l'objet pdfplumber
            if pdf_obj is not None:
                try:
                    # Récupération propre du nom du fichier si disponible
                    if hasattr(pdf_obj, "stream") and hasattr(pdf_obj.stream, "name"):
                        contract_id = os.path.basename(pdf_obj.stream.name)
                    
                    # Extraction du texte de toutes les pages du PDF
                    pages_text = [page.extract_text() for page in pdf_obj.pages if page.extract_text()]
                    full_text = "\n".join(pages_text)
                except Exception as e:
                    full_text = f"Erreur d'extraction du texte: {str(e)}"

            extracted_records.append({
                "contract_id": contract_id,
                "clause_type": "Legal Contract Full Text",
                "context": full_text,  # Le texte complet du contrat pour le RAG
                "ground_truth": "N/A",
                "has_answer": True,
            })

        # Conversion en DataFrame et sauvegarde CSV
        df = pd.DataFrame(extracted_records)
        output_path = os.path.join(self.processed_data_dir, "cuad_processed.csv")
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"💾 Données traitées sauvegardées avec succès dans {output_path} ({len(df)} lignes).")
        return df


if __name__ == "__main__":
    loader = CUADDataLoader()
    df_processed = loader.process_and_extract_clauses(limit=5)
    print("\n📊 Aperçu des données extraites (avec noms de fichiers propres) :")
    print(df_processed[["contract_id", "clause_type", "has_answer"]].head())
    print(f"\n📝 Longueur du texte extrait pour le 1er contrat : {len(df_processed.iloc[0]['context'])} caractères.")