import streamlit as st
import os
import tempfile
import shutil
from dotenv import load_dotenv
from agent import build_graph, AgentState
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

# ---- SESSION STATE ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "app" not in st.session_state:
    st.session_state.app = build_graph()
if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

# ---- STREAMING FINAL ANSWER ----
def stream_final_answer(messages, system_prompt):
    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system=system_prompt,
        messages=messages
    ) as stream:
        for text in stream.text_stream:
            yield text

# ---- SIDEBAR ----
with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Drop PDFs here to analyze",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        new_filenames = sorted([f.name for f in uploaded_files])

        if new_filenames != st.session_state.uploaded_filenames:
            pdf_paths = []
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_paths.append(temp_path)

            with st.spinner(f"Ingesting {len(pdf_paths)} document(s)..."):
                if os.path.exists(".chromadb"):
                    shutil.rmtree(".chromadb")
                st.session_state.collection = build_vector_store(pdf_paths)
                st.session_state.uploaded_filenames = new_filenames
                st.session_state.messages = []
                st.session_state.conversation_history = []

            st.success(f"Ready — {len(pdf_paths)} document(s) loaded")

        st.divider()
        for filename in st.session_state.uploaded_filenames:
            st.markdown(f"📄 `{filename}`")

    else:
        st.info("Upload one or more PDFs to get started")

    st.divider()
    st.header("About")
    st.markdown("""
    This agent:
    - Retrieves relevant chunks from your PDFs
    - Evaluates retrieval quality and retries if needed
    - Fetches live market data when relevant
    - Cites sources for every answer
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
if st.session_state.collection is None:
    st.warning("👈 Upload PDFs in the sidebar to get started")
else:
    if question := st.chat_input("Ask a question about your documents..."):

        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.conversation_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):

            # ---- PHASE 1: Run agent for retrieval and tool use ----
            # Show status updates as agent works
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
                st.write("Retrieving relevant documents...")
                final_state = st.session_state.app.invoke(initial_state)
                # DEBUG
                import json
                last = final_state["messages"][-1]
                print(f"DEBUG last message role: {last['role']}")
                print(f"DEBUG last message content type: {type(last['content'])}")
                print(f"DEBUG last message content: {last['content']}")
                if final_state.get("retry_count", 0) > 0:
                    st.write(f"🔄 Refined retrieval {final_state['retry_count']} time(s)")

                st.write("✅ Context ready — generating answer...")

            status.update(label="Research complete", state="complete")

            # ---- PHASE 2: Stream the final answer ----
            # Build messages up to but not including final answer
            # so we can stream it fresh
           # ---- PHASE 2: Extract and display answer ----
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
            # Simulate streaming by writing chunk by chunk
            def stream_text(text):
                import time
                chunk_size = 15
                for i in range(0, len(text), chunk_size):
                    yield text[i:i + chunk_size]
                    time.sleep(0.01)

            st.write_stream(stream_text(answer))

            # ---- SOURCES ----
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