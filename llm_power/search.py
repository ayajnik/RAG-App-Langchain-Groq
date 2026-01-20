try:
    from dotenv import load_dotenv
    load_dotenv()
    import os
    from vector_store.vectorstore import vectorstore
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq
    from google.genai.errors import ClientError
except ImportError as e:
    print(f"ImportError: {e}. Please ensure all required packages are installed.")

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", gemini_model: str = "gemini-1.5-flash"):
        self.vs = vectorstore(persist_dir)
        self.vector_db = self.vs.load()
        self.vs.load()
        
        # Initialize Gemini (primary LLM)
        self.gemini_llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            temperature=0.7
        )
        
        # Initialize Groq (fallback LLM)
        self.groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.current_llm = self.gemini_llm
        self.use_groq = False
        print(f"RAGSearch initialized with primary LLM: {gemini_model}, fallback: Groq (llama-3.3-70b-versatile)")

    def search_and_summarize(self, query: str, top_k: int):
        results = self.vs.query(query, top_k=top_k)
        print(f"Search results for query '{query}': {results}")
        texts = [r["metadata"].get("text"," ") for r in results if r["metadata"]]
        context = "\n".join(texts)
        prompt = f"Summarize the following information in a concise manner:\n\n{context}\n\nSummary:"
        
        try:
            summary = self.gemini_llm.invoke(prompt)
            self.use_groq = False
            return summary.content
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "rate" in str(e).lower():
                print(f" Gemini rate limit exceeded: {e}")
                print("Switching to Groq...")
                self.use_groq = True
                try:
                    summary = self.groq_llm.invoke(prompt)
                    return summary.content
                except Exception as groq_error:
                    print(f"Groq also failed: {groq_error}")
                    raise
            else:
                print(f"Gemini error: {e}")
                raise