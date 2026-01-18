try:
    import os
    from pathlib import Path
    from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("All required libraries for processing data are successfully imported.")
except ImportError as e:
    raise ImportError("Required libraries are not installed. Please install langchain and langchain-community packages.") from e

class DataProcessor:

    def __init__(self, pdf_directory):
        self.pdf_directory = pdf_directory
        self.all_documents = []

    def read_pdf_path(self):
        pdf_files = Path(self.pdf_directory)
        return pdf_files
    
    def get_pdf_files(self):

        files = self.read_pdf_path()
        pdf_files_present = list(files.glob("**/*.pdf"))
        print("Total PDF files found:", len(pdf_files_present))
        return pdf_files_present
    
    def load_pdf_files(self):

        pdf_files_found = self.get_pdf_files()
        for files in pdf_files_found:
            try:
                print("Loading file:", files)
                loader = PyPDFLoader(str(files))
                documents = loader.load()
                print(f"Loaded {len(documents)} documents from {files}")
                self.all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading {files}: {e}")
        print("Total documents loaded:", len(self.all_documents))
        return self.all_documents

    

# if __name__ == "__main__":
#     pdf_directory = str(Path(__file__).parent.parent / "data")
#     print("PDF Directory:", pdf_directory)
#     processor = DataProcessor(pdf_directory)
#     documents = processor.load_pdf_files()


