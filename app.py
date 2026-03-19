import streamlit as st
import os
import time
from dotenv import load_dotenv
from agent import build_graph
from rag import build_vector_store
import anthropic

load_dotenv()

client = anthropic.Anthropic()

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Financial Research Agent")
        st.markdown("""
        An agentic RAG system for financial document research with live market data integration.
        
        Built with Claude, LangGraph, ChromaDB, and Streamlit.
        """)
        st.link_button(
            "View README & Architecture on GitHub",
            "https://github.com/yourusername/financial-research-agent"
        )
        st.divider()
        st.markdown("**Demo Access**")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if password == os.environ.get("APP_PASSWORD"):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()

check_password()

# ---- PAGE SETUP ----
st.set_page_config(
    page_title="Financial Research Agent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial Research Agent")
st.caption("Intelligent document research agent with live market data")

# ---- CONFIG ----
PDF_PATHS = ["nvidia_10k.pdf", "apple_10k.pdf", "meta-10k.pdf"]
COMPANIES = ["Nvidia", "Apple", "Meta"]

# ---- SESSION STATE ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "app" not in st.session_state:
    st.session_state.app = build_graph()
if "collection" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.collection = build_vector_store([])

# ---- SIDEBAR ----
with st.sidebar:
    st.header("Loaded Documents")
    for company, path in zip(COMPANIES, PDF_PATHS):
        st.markdown(f"📄 **{company}** 10-K")

    st.divider()
    st.header("Try asking...")
    st.markdown("""
    - What was Nvidia's revenue growth YoY?
    - Compare Apple and Meta's risk factors
    - What is Nvidia's current PE ratio?
    - Where is Meta investing most heavily?
    - Compare gross margins across all three companies
    """)

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

# ---- RENDER CHAT HISTORY ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- SYSTEM PROMPT ----
SYSTEM_PROMPT = """You are a senior hedge fund researcher with access to tools.

IMPORTANT RULES:
- You MUST use search_documents for ANY question about company financials, filings, or business information. Never answer from memory.
- You MUST use get_stock_price for ANY question about current price, valuation, or market data.
- Always call tools before forming an answer. Do not respond with text until you have retrieved real data.
- Be concise and signal focused in your final answer."""

# ---- HANDLE INPUT ----
if question := st.chat_input("Ask about Nvidia, Apple, or Meta..."):

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.conversation_history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        status = st.status("Researching...", expanded=False)

        initial_state = {
            "messages": st.session_state.conversation_history.copy(),
            "collection": st.session_state.collection,
            "question": question,
            "retrieved_chunks": [],
            "retrieval_sufficient": False,
            "reformulated_query": None,
            "retry_count": 0,
            "stop_reason": None
        }

        with status:
            st.write("🔍 Retrieving relevant documents...")
            final_state = st.session_state.app.invoke(initial_state)

            if final_state.get("retry_count", 0) > 0:
                st.write(f"🔄 Refined retrieval {final_state['retry_count']} time(s)")

            st.write("✅ Context ready — generating answer...")

        status.update(label="Research complete", state="complete")

        answer = None
        last_message = final_state["messages"][-1]
        if isinstance(last_message["content"], list):
            for block in last_message["content"]:
                if hasattr(block, "text"):
                    answer = block.text
                    break
        else:
            answer = last_message["content"]

        if answer:
            def stream_text(text):
                chunk_size = 15
                for i in range(0, len(text), chunk_size):
                    yield text[i:i + chunk_size]
                    time.sleep(0.01)

            st.write_stream(stream_text(answer))

            if final_state.get("retrieved_chunks"):
                with st.expander("📚 Sources"):
                    for chunk in final_state["retrieved_chunks"]:
                        st.caption(f"**{chunk['source']}**, Page {chunk['page']}")
                        st.text(chunk["text"][:300] + "...")
                        st.divider()

    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": answer
        })