from data_ingestion.process_data import DataProcessor
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

class EmbeddingGenerator:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500, chunk_overlap:int = 200):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)

    def chunk_documents(self, documents: List[Any]) -> List[Any]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ",""]
        )
        chunks = splitter.split_documents(documents)
        print(f"Total chunks created: {len(chunks)}")
        return chunks
    
    def embed_chunks(self,chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
# if __name__ == "__main__":
#     pdf_directory = str(Path(__file__).parent.parent / "data")
#     print("PDF Directory:", pdf_directory)
#     processor = DataProcessor(pdf_directory)
#     documents = processor.load_pdf_files()

#     embedder = EmbeddingGenerator()
#     chunks = embedder.chunk_documents(documents)
#     embeddings = embedder.embed_chunks(chunks)
#     print("Embeddings shape:", embeddings.shape)

