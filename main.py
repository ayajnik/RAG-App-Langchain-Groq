from data_ingestion.process_data import DataProcessor
from data_ingestion.embedding import EmbeddingGenerator
from vector_store.vectorstore import vectorstore
from llm_power.search import RAGSearch
from pathlib import Path

pdf_directory = str(Path(__file__).parent / "data")
print("PDF Directory:", pdf_directory)
processor = DataProcessor(pdf_directory)
documents = processor.load_pdf_files()

embedder = EmbeddingGenerator()
chunks = embedder.chunk_documents(documents)
embeddings = embedder.embed_chunks(chunks)
print("Embeddings shape:", embeddings.shape)

vs = vectorstore("faiss_store")
#vs.add_embeddings(embeddings=embeddings)
vs.build_from_documents(chunks, embeddings)
vector_db=vs.load()
print(vs.query("How Old is India", top_k=3))

rag_search = RAGSearch()
query = "Why do we call India an ancient civilization?"
summary = rag_search.search_and_summarize(query, top_k=3)
print("Summary:", summary)