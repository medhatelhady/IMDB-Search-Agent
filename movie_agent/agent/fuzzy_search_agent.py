"""Fuzzy Search Agent: finds movies by approximate title matching using pg_trgm."""

import os
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from movie_agent.search.fuzzy import fuzzy_search_title
from movie_agent.agent.prompts import FUZZY_SEARCH_AGENT_PROMPT


# ============================================================
# Tool definition
# ============================================================
@tool
def search_movie_by_title(title: str, top_k: int = 5) -> str:
    """Search for movies by title using fuzzy matching.

    Use this tool when the user provides a movie title (or something close to it)
    and you need to find the actual movie. Handles typos, partial names, and
    approximate matches.

    Args:
        title: The movie title to search for (can be misspelled or partial).
        top_k: Number of results to return (default 5, max 10).

    Returns:
        Formatted string with matching movies and their similarity scores.
    """
    top_k = min(max(top_k, 1), 10)
    results = fuzzy_search_title(title, top_k=top_k)

    if not results:
        return f"No movies found matching the title '{title}'. Try a different spelling or a more complete title."

    output_lines = []
    for i, movie in enumerate(results, 1):
        overview = movie["overview"] or "No overview available."
        overview_short = overview[:150] + "..." if len(overview) > 150 else overview
        output_lines.append(
            f"{i}. **{movie['title']}** (match: {movie['similarity']})\n"
            f"   Rating: {movie['vote_average']} | Released: {movie['release_date']}\n"
            f"   {overview_short}"
        )

    return "\n\n".join(output_lines)


# ============================================================
# Agent construction
# ============================================================
def _build_agent():
    """Build and return the fuzzy search agent."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        base_url="https://openai.vocareum.com/v1",
        api_key="voc-46640206319004520369916a7f3655c69c53.81266809",
    )

    tools = [search_movie_by_title]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=FUZZY_SEARCH_AGENT_PROMPT,
        checkpointer=InMemorySaver(),
    )
    return agent


# Lazy singleton
_agent = None


def get_agent():
    """Get or create the fuzzy search agent singleton."""
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def ask(question: str, session_id: str | None = None) -> dict:
    """Ask the fuzzy search agent to find a movie by title.

    Args:
        question: The user's query mentioning a movie title.
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
        for node_name, node_output in chunk.items():
            messages = node_output.get("messages", [])
            for message in messages:
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        yield {
                            "type": "tool_call",
                            "tool": tool_call["name"],
                            "input": str(tool_call.get("args", {})),
                            "session_id": session_id,
                        }

                elif message.type == "tool":
                    yield {
                        "type": "tool_result",
                        "tool": getattr(message, "name", "unknown"),
                        "output": message.content[:2000],
                        "session_id": session_id,
                    }

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
