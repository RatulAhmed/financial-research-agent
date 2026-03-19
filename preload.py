import os
os.environ['CHROMA_CACHE_DIR'] = '/root/.cache/chroma'
from chromadb.utils import embedding_functions
ef = embedding_functions.DefaultEmbeddingFunction()
ef(['test document'])
print("Embedding model pre-cached successfully")