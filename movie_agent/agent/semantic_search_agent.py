"""Semantic Search Agent: finds movies by meaning/description using vector similarity."""

import os
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from movie_agent.config import DATABASE_URL
from movie_agent.search.semantic import semantic_search
from movie_agent.agent.prompts import SEMANTIC_SEARCH_AGENT_PROMPT


# ============================================================
# Tool definition
# ============================================================
@tool
def search_movies_by_description(query: str, top_k: int = 5) -> str:
    """Search for movies by describing what they're about.

    Use this tool when the user describes a type of movie, a plot,
    a theme, or a mood they're looking for. The search uses semantic
    similarity against movie overviews/descriptions.

    Args:
        query: Natural language description of the kind of movie
               (e.g. "a heist movie with a clever twist",
                "animated movie about feelings and emotions",
                "dark sci-fi dystopia").
        top_k: Number of results to return (default 5, max 20).

    Returns:
        Formatted string with matching movies and their details.
    """
    top_k = min(max(top_k, 1), 20)
    results = semantic_search(query, top_k=top_k)

    if not results:
        return "No movies found matching that description."

    output_lines = []
    for i, movie in enumerate(results, 1):
        overview = movie["overview"] or "No overview available."
        overview_short = overview[:200] + "..." if len(overview) > 200 else overview
        output_lines.append(
            f"{i}. **{movie['title']}** (similarity: {movie['similarity']})\n"
            f"   Rating: {movie['vote_average']} | Released: {movie['release_date']}\n"
            f"   {overview_short}"
        )

    return "\n\n".join(output_lines)


# ============================================================
# Agent construction
# ============================================================
def _build_agent():
    """Build and return the semantic search agent."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        base_url="https://openai.vocareum.com/v1",
        api_key="voc-46640206319004520369916a7f3655c69c53.81266809",
    )

    tools = [search_movies_by_description]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SEMANTIC_SEARCH_AGENT_PROMPT,
        checkpointer=InMemorySaver(),
    )
    return agent


# Lazy singleton
_agent = None


def get_agent():
    """Get or create the semantic search agent singleton."""
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def ask(question: str, session_id: str | None = None) -> dict:
    """Ask the semantic search agent a question.

    Args:
        question: What kind of movie the user is looking for.
        session_id: Optional session ID for conversation continuity.

    Returns:
        dict with 'answer' and 'session_id'.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    agent = get_agent()
    config = {"configurable": {"thread_id": session_id}}

    result = agent.invoke({"messages": [("human", question)]}, config=config)
    answer = result["messages"][-1].content

    return {"answer": answer, "session_id": session_id}


def ask_stream(question: str, session_id: str | None = None):
    """Stream all agent steps: tool calls, tool results, and answer tokens.

    Yields:
        dict with one of:
          - {"type": "tool_call", "tool": str, "input": str, "session_id": str}
          - {"type": "tool_result", "tool": str, "output": str, "session_id": str}
          - {"type": "token", "content": str, "session_id": str}
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    agent = get_agent()
    config = {"configurable": {"thread_id": session_id}}

    for chunk in agent.stream(
        {"messages": [("human", question)]},
        config=config,
        stream_mode="updates",
    ):
        # Each chunk is {node_name: {messages: [...]}}
        for node_name, node_output in chunk.items():
            messages = node_output.get("messages", [])
            for message in messages:
                # Agent node emitting tool calls
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        yield {
                            "type": "tool_call",
                            "tool": tool_call["name"],
                            "input": str(tool_call.get("args", {})),
                            "session_id": session_id,
                        }
                    # If the message also has text content alongside tool calls, skip it
                    # (it's usually empty or just reasoning)

                # Tool node returning results
                elif message.type == "tool":
                    yield {
                        "type": "tool_result",
                        "tool": getattr(message, "name", "unknown"),
                        "output": message.content[:2000],
                        "session_id": session_id,
                    }

                # Agent node emitting final answer (no tool calls)
                elif (
                    hasattr(message, "content")
                    and message.content
                    and node_name == "agent"
                    and not getattr(message, "tool_calls", None)
                ):
                    yield {
                        "type": "token",
                        "content": message.content,
                        "session_id": session_id,
                    }
