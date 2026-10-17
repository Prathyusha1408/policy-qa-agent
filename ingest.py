"""Step 1: PDFs -> chunks -> OpenAI embeddings -> ChromaDB (persisted on disk)."""
import shutil
 
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
import config
 
 
def _reset_index() -> None:
    """Remove the old index. On Windows/OneDrive the folder can be locked, so fall back
    to deleting the Chroma collection instead of the files."""
    if not config.PERSIST_DIR.exists():
        return
    try:
        shutil.rmtree(config.PERSIST_DIR)
    except PermissionError:
        print("chroma_db folder is locked; clearing the collection instead...")
        Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
            persist_directory=str(config.PERSIST_DIR),
        ).delete_collection()
 
 
def build_index() -> int:
    """Rebuild the vector index from every PDF in data/. Returns the chunk count."""
    config.require_api_key()
    pdf_paths = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {config.DATA_DIR}. Add some first.")
 
    documents = []
    for path in pdf_paths:
        pages = PyPDFLoader(str(path)).load()
        print(f"Loaded {path.name}: {len(pages)} pages")
        documents.extend(pages)
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
 
    # Start clean so re-indexing never creates duplicates
    _reset_index()
 
    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.PERSIST_DIR),
    )
    print(f"Index saved to {config.PERSIST_DIR}")
    return len(chunks)
 
 
if __name__ == "__main__":
    build_index()
 