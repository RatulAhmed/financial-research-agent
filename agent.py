import os
import json
from dotenv import load_dotenv
import anthropic
import yfinance as yf
from typing import TypedDict
from langgraph.graph import StateGraph, END
from rag import build_vector_store, retrieve

load_dotenv()

client = anthropic.Anthropic()

# ---- STATE ----
class AgentState(TypedDict):
    messages: list
    collection: object

# ---- TOOL FUNCTIONS ----
def get_stock_price(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "ticker": ticker,
        "price": info.get("currentPrice"),
        "pe_ratio": info.get("trailingPE"),
        "market_cap": info.get("marketCap"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow")
    }

def search_documents(query: str, collection) -> dict:
    results = retrieve(collection, query, n=3)
    context_parts = []
    for doc, meta in results:
        context_parts.append(f"[{meta['source']}, Page {meta['page']}]: {doc}")
    return {"results": context_parts}

# ---- TOOL DEFINITIONS ----
TOOLS = [
    {
        "name": "get_stock_price",
        "description": "Get current stock price and key metrics for a ticker. Use for current market data and valuation metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol e.g. NVDA, AAPL"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "search_documents",
        "description": "Search financial document knowledge base for information from 10-K filings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant document chunks"
                }
            },
            "required": ["query"]
        }
    }
]

# ---- NODES ----
def call_claude(state: AgentState) -> AgentState:
    print("\n[NODE] call_claude")
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system="""You are a senior hedge fund researcher with access to tools.

IMPORTANT RULES:
- You MUST use search_documents for ANY question about company financials, filings, or business information. Never answer from memory.
- You MUST use get_stock_price for ANY question about current price, valuation, or market data.
- Always call tools before forming an answer. Do not respond with text until you have retrieved real data.
- Be concise and signal focused in your final answer.""",
        tools=TOOLS,
        messages=state["messages"]
    )

    # DEBUG - see exactly what Claude returned
    print(f"  stop_reason: {response.stop_reason}")
    print(f"  content blocks: {response.content}")

    state["messages"].append({
        "role": "assistant",
        "content": response.content
    })
    state["stop_reason"] = response.stop_reason
    return state

def execute_tools(state: AgentState) -> AgentState:
    print("\n[NODE] execute_tools")
    last_message = state["messages"][-1]
    tool_results = []

    for block in last_message["content"]:
        # Handle Anthropic SDK objects, not just dicts
        block_type = block.type if hasattr(block, "type") else block.get("type")
        
        if block_type == "tool_use":
            name = block.name if hasattr(block, "name") else block.get("name")
            input_data = block.input if hasattr(block, "input") else block.get("input")
            block_id = block.id if hasattr(block, "id") else block.get("id")
            
            print(f"  [TOOL CALL] {name}({input_data})")
            
            if name == "get_stock_price":
                result = get_stock_price(input_data["ticker"])
            elif name == "search_documents":
                result = search_documents(input_data["query"], state["collection"])
            else:
                result = {"error": f"Unknown tool: {name}"}
            
            print(f"  [TOOL RESULT] {json.dumps(result)[:150]}...")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block_id,
                "content": json.dumps(result)
            })

    state["messages"].append({
        "role": "user",
        "content": tool_results
    })
    return state

# ---- CONDITIONAL EDGE ----
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    
    # Check if any block in the last message is a tool use
    for block in last_message["content"]:
        block_type = block.type if hasattr(block, "type") else block.get("type")
        if block_type == "tool_use":
            return "execute_tools"
    
    return END

# ---- BUILD THE GRAPH ----
def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("call_claude", call_claude)
    graph.add_node("execute_tools", execute_tools)

    # Set entry point
    graph.set_entry_point("call_claude")

    # Add edges
    graph.add_conditional_edges("call_claude", should_continue)
    graph.add_edge("execute_tools", "call_claude")

    return graph.compile()

# ---- MAIN ----
if __name__ == "__main__":
    import sys
    pdf_paths = sys.argv[1:]
    if not pdf_paths:
        print("Usage: uv run agent.py file1.pdf file2.pdf ...")
        sys.exit(1)

    collection = build_vector_store(pdf_paths)
    app = build_graph()

    # Maintain conversation history across questions
    conversation_history = []

    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == "quit":
            break

        conversation_history.append({"role": "user", "content": question})

        initial_state = {
            "messages": conversation_history.copy(),
            "collection": collection,
            "stop_reason": None
        }

        final_state = app.invoke(initial_state)
        
        # Extract final answer
        last_message = final_state["messages"][-1]
        answer = None
        if isinstance(last_message["content"], list):
            for block in last_message["content"]:
                if hasattr(block, "text"):
                    answer = block.text
        else:
            answer = last_message["content"]

        if answer:
            print(f"\nAnswer: {answer}")
            # Add assistant response to history
            conversation_history.append({
                "role": "assistant",
                "content": answer
            })