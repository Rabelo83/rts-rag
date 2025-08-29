# ingest.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = Path("data")
DB_DIR = "db"

def load_all_pdfs(folder: Path):
    docs = []
    for pdf in sorted(folder.glob("*.pdf")):
        print(f"[INGEST] Loading {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
    return docs

def main():
    # 1) Load PDFs
    docs = load_all_pdfs(DATA_DIR)
    if not docs:
        print("No PDFs found in ./data. Add a PDF and rerun.")
        return

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(docs)
    print(f"[INGEST] Split into {len(splits)} chunks")

    # 3) Embed with Ollama (local)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4) Store in persistent Chroma DB
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    vectordb.persist()
    print(f"[DONE] Indexed {len(splits)} chunks into {DB_DIR}/")

if __name__ == "__main__":
    main()
