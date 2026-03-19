import os
from dotenv import load_dotenv
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import hashlib

load_dotenv()

client = anthropic.Anthropic()

# ---- STEP 1: Load and chunk a PDF ----
def load_pdf(path, chunk_size=500, overlap=50):
    reader = PdfReader(path)
    filename = os.path.basename(path)
    
    # First extract all text with page tracking
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            pages.append({"page": i + 1, "text": text})

    # Combine all text but track where each page starts
    chunks = []
    chunk_id = 0

    for page_data in pages:
        words = page_data["text"].split()
        page_num = page_data["page"]

        # Slide a window of chunk_size words with overlap
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "id": f"{filename}_chunk_{chunk_id}",
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "page": page_num,
                    "chunk": chunk_id,
                    "start_word": start,
                }
            })
            chunk_id += 1
            start += chunk_size - overlap  # overlap slides back

    return chunks

# ---- STEP 2: Get a unique hash for a PDF ----
def get_pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# ---- STEP 3: Build/load persistent vector store ----
def build_vector_store(pdf_paths):
    chroma = chromadb.PersistentClient(path=".chromadb")
    collection = chroma.get_or_create_collection("rag_demo")

    # Only ingest if empty
    if collection.count() == 0:
        for path in pdf_paths:
            filename = os.path.basename(path)
            print(f"Ingesting {filename}...")
            chunks = load_pdf(path)
            collection.add(
                documents=[c["text"] for c in chunks],
                ids=[c["id"] for c in chunks],
                metadatas=[c["metadata"] for c in chunks]
            )
            print(f"  → {len(chunks)} chunks ingested")
        print(f"\nVector store ready ({collection.count()} total chunks)\n")
    else:
        print(f"Loaded existing vector store ({collection.count()} chunks)")

    return collection

# ---- STEP 4: Retrieve relevant chunks ----
def retrieve(collection, query, n=3):
    results = collection.query(query_texts=[query], n_results=n)
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    return list(zip(docs, metadatas))

# ---- STEP 5: Ask Claude ----
SYSTEM_PROMPT = """You are a senior researcher at a quantitative hedge fund. 
Your job is to extract actionable signals and insights from financial documents.

Your rules:
- Be concise and direct. No fluff. Every sentence must carry information.
- Lead with the most important insight first.
- Flag risks, inflection points, and surprises — these matter more than confirming consensus.
- When comparing companies, focus on relative positioning and competitive dynamics.
- If the documents don't contain enough information to answer confidently, say so explicitly. Never speculate or hallucinate.
- Format responses with brief headers when covering multiple angles.
- Always note which document and page your key claims come from.

You are talking to a portfolio manager who has 30 seconds to read your answer."""

def ask(collection, question, debug=False):
    results = retrieve(collection, question)
    
    context_parts = []
    sources = []
    for doc, meta in results:
        context_parts.append(f"[Source: {meta['source']}, Page {meta['page']}]\n{doc}")
        sources.append(f"  - {meta['source']}, Page {meta['page']}")
    
    context_text = "\n\n---\n\n".join(context_parts)

    prompt = f"""Answer based only on the context below.

Context:
{context_text}

Question: {question}"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.content[0].text
    unique_sources = sorted(set(sources))
    
    return answer, unique_sources


# ---- MAIN ----
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run rag.py file1.pdf file2.pdf ...")
        sys.exit(1)

    pdf_paths = sys.argv[1:]
    collection = build_vector_store(pdf_paths)

    while True:
        question = input("Ask a question (or 'quit'): ")
        if question.lower() == "quit":
            break
        answer, sources = ask(collection, question, debug=False)
        print(f"\nAnswer: {answer}")
        print(f"\nSources:")
        for s in sources:
            print(s)
        print()