from data_ingestion.process_data import DataProcessor
from data_ingestion.embedding import EmbeddingGenerator
from pathlib import Path

pdf_directory = str(Path(__file__).parent / "data")
print("PDF Directory:", pdf_directory)
processor = DataProcessor(pdf_directory)
documents = processor.load_pdf_files()

embedder = EmbeddingGenerator()
chunks = embedder.chunk_documents(documents)
embeddings = embedder.embed_chunks(chunks)
print("Embeddings shape:", embeddings.shape)