import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


class LegalRAGIndexer:
    """Gère le découpage (chunking), la vectorisation et l'indexation

    des contrats dans une base vectorielle ChromaDB.
    """

    def __init__(
        self,
        csv_path: str = "data/processed/cuad_processed.csv",
        db_dir: str = "data/vector_db",
    ):
        self.csv_path = csv_path
        self.db_dir = db_dir
        
        # Modèle d'embedding open-source performant et léger
        print("🔄 Chargement du modèle d'embedding (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialisation de la base vectorielle locale ChromaDB
        print(f"🗄️ Initialisation de la base vectorielle dans {self.db_dir}...")
        self.chroma_client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.chroma_client.get_or_create_collection(name="cuad_contracts")

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """Découpe un texte en morceaux (chunks) avec chevauchement."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap  # Recul pour l'overlap

        return chunks

    def build_index(self, chunk_size: int = 1000, overlap: int = 200):
        """Lit le CSV traité, découpe les contrats, vectorise et stocke dans ChromaDB."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Le fichier {self.csv_path} est introuvable. Exécutez data_loader.py d'abord !")

        print(f"📖 Lecture des données depuis {self.csv_path}...")
        df = pd.read_csv(self.csv_path)

        documents = []
        metadatas = []
        ids = []

        chunk_counter = 0

        print("⚙️ Découpage et préparation des chunks...")
        for idx, row in df.iterrows():
            contract_id = row["contract_id"]
            full_context = row["context"]

            # Découpage du texte du contrat
            chunks = self.chunk_text(full_context, chunk_size=chunk_size, overlap=overlap)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{contract_id}_chunk_{chunk_idx}"
                documents.append(chunk)
                metadatas.append({"contract_id": contract_id, "chunk_index": chunk_idx})
                ids.append(chunk_id)
                chunk_counter += 1

        print(f"📦 Total de {chunk_counter} chunks générés à partir de {len(df)} contrats.")
        
        if chunk_counter == 0:
            print("⚠️ Aucun texte à indexer.")
            return

        print("🔄 Génération des embeddings et insertion dans ChromaDB (cela peut prendre un instant)...")
        # Génération des embeddings par batch pour plus d'efficacité
        embeddings = self.model.encode(documents, show_progress_bar=True).tolist()

        # Ajout dans la base vectorielle
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✅ Indexation réussie ! {chunk_counter} vecteurs stockés dans ChromaDB.")


if __name__ == "__main__":
    indexer = LegalRAGIndexer()
    indexer.build_index()