import os
os.environ['CHROMA_CACHE_DIR'] = '/root/.cache/chroma'
from chromadb.utils import embedding_functions
from rag import build_vector_store

# Pre-download embedding model
ef = embedding_functions.DefaultEmbeddingFunction()
ef(['test document'])
print("Embedding model cached")

# Pre-build vector store from PDFs
PDF_PATHS = ["nvidia_10k.pdf", "apple_10k.pdf", "meta-10k.pdf"]
build_vector_store(PDF_PATHS)
print("Vector store pre-built")