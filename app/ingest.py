import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config


def get_vectorstore():
    return Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=OpenAIEmbeddings(model=config.EMBED_MODEL),
    )

def ingest(pdf_dir="data/pdfs"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    pages = []
    for pdf in Path(pdf_dir).glob("*.pdf"):
        for page in PyPDFLoader(str(pdf)).load():
            page.metadata["source"] = pdf.name   # keep 'page' from the loader
            pages.append(page)

    chunks = splitter.split_documents(pages)
    # Deterministic IDs: re-running ingestion updates instead of duplicating
    ids = [hashlib.md5(
        f"{c.metadata['source']}-{c.metadata['page']}-{c.page_content}".encode()
    ).hexdigest() for c in chunks]

    get_vectorstore().add_documents(chunks, ids=ids)
    print(f"Ingested {len(chunks)} chunks from {len(pages)} pages")

if __name__ == "__main__":
    ingest()