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
        self.vs.load()
        self.llm = ChatGoogleGenerativeAI(model_name=llm_model)
        print(f"RAGSearch initialized with LLM model: {llm_model} and embedding model: {embedding_model}")

    def search_and_summarize(self, query: str, top_k: int):
        results = self.vs.query(query, top_k=top_k)
        print(f"Search results for query '{query}': {results}")
        texts = [r["metadata"].get("text"," ") for r in results if r["metadata"]]
        context = "\n".join(texts)
        prompt = f"Summarize the following information in a concise manner:\n\n{context}\n\nSummary:"
        summary = self.llm.invoke(prompt)
        return summary.text

    