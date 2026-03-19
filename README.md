# Financial Research Agent

A RAG-based research agent for analyzing SEC 10-K filings. Ask questions about Nvidia, Apple, and Meta — the agent retrieves relevant context from their annual filings, cross-references live market data, and synthesizes a concise answer.

## How it works

The agent is built on a LangGraph graph with two core loops. The first handles tool orchestration — Claude decides whether to search the document knowledge base, fetch live market data, or both, depending on the question. The second is a retrieval evaluation loop — after pulling document chunks, the agent judges whether the context is actually sufficient to answer the question. If not, it reformulates the query and tries again before generating a response.

Documents are chunked and embedded using ChromaDB with a sentence transformer model. Retrieval is semantic — the agent finds chunks by meaning, not keyword matching.

The stack is Python, Anthropic Claude, LangGraph, ChromaDB, and Streamlit.

## Architecture
```
User question
    ↓
LangGraph agent
    ↓
Tool selection (Claude)
    ├── search_documents → ChromaDB semantic search → 10-K chunks
    └── get_stock_price → Yahoo Finance → live market data
    ↓
Retrieval evaluation
    ├── Sufficient → generate answer
    └── Insufficient → reformulate query → search again
    ↓
Answer with citations
```

## Running locally
```bash
git clone https://github.com/yourusername/financial-research-agent
cd financial-research-agent
uv sync
cp .env.example .env  # add your ANTHROPIC_API_KEY
uv run streamlit run app.py
```

## Tech stack

- [Anthropic Claude](https://anthropic.com) — reasoning and answer generation
- [LangGraph](https://langchain-ai.github.io/langgraph/) — agent orchestration
- [ChromaDB](https://trychroma.com) — vector storage and semantic search
- [Streamlit](https://streamlit.io) — frontend
- [yfinance](https://pypi.org/project/yfinance/) — live market data

## Notes

The original implementation supported drag-and-drop PDF uploads so any document could be analyzed at runtime. This was removed in the deployed version due to server timeout constraints — embedding large PDFs (400+ pages) during a live request exceeds Railway's free tier limits. The upload feature works fine locally and the code for it lives in the `experiments` branch. The deployed version ships with Nvidia, Apple, and Meta 10-Ks pre-embedded into the Docker image at build time.

## example .env
ANTHROPIC_API_KEY=your-key-here
APP_PASSWORD=your-password-here

