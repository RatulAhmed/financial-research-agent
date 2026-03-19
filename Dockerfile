FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the embedding model during build so it's baked into the image
RUN python -c "from chromadb.utils import embedding_functions; embedding_functions.DefaultEmbeddingFunction()"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port $PORT" , "--server.address", "0.0.0.0"]