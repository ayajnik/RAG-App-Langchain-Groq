try:
    from dotenv import load_dotenv
    load_dotenv()
    import os
    from vector_store.vectorstore import vectorstore
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:
    print(f"ImportError: {e}. Please ensure all required packages are installed.")

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gemini-pro"):
        self.vs = vectorstore(persist_dir)
        self.vector_db = self.vs.load()
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

    