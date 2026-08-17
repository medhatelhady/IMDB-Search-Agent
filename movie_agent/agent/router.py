"""Router Agent: decides whether to use the SQL agent or the Semantic Search agent
based on the user's query, then delegates and returns the result.
"""

import os
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from movie_agent.agent.sql_agent import ask as sql_ask
from movie_agent.agent.semantic_search_agent import ask as semantic_ask
from movie_agent.agent.fuzzy_search_agent import ask as fuzzy_ask
from movie_agent.agent.prompts import ROUTER_AGENT_PROMPT

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
# ============================================================
# Tools that wrap the sub-agents
# ============================================================
@tool
def sql_agent_tool(question: str) -> str:
    """Use this tool for structured/factual questions about the movies database.

    Best for questions involving:
    - Specific numbers, counts, averages, rankings, comparisons
    - Filtering by exact values (year, rating, budget, revenue, runtime)
    - Questions about genres, production companies, countries, or languages
    - "How many...", "What is the highest/lowest...", "List all movies where..."
    - Any question that requires precise data from database columns

    Args:
        question: The user's question to answer using SQL queries.

    Returns:
        The answer from the SQL agent.
    """
    result = sql_ask(question)
    return result["answer"]


@tool
def semantic_search_tool(query: str) -> str:
    """Use this tool for descriptive/thematic movie searches based on plot, mood, or theme.

    Best for questions involving:
    - Finding movies by describing a plot or story ("a movie about...")
    - Mood or theme-based recommendations ("something dark and mysterious")
    - Finding similar movies to a description
    - Vague or creative queries that can't be answered with exact database filters
    - "Recommend me...", "Find movies like...", "I'm looking for..."

    Args:
        query: A natural language description of what kind of movie the user wants.

    Returns:
        The answer from the semantic search agent.
    """
    result = semantic_ask(query)
    return result["answer"]


@tool
def fuzzy_title_search_tool(title: str) -> str:
    """Use this tool when the user mentions a specific movie title (or something close to it).

    Best for questions involving:
    - Looking up a specific movie by name ("Tell me about Interstellar")
    - Misspelled or partial titles ("What's the movie Incpetion about?", "the dark nite")
    - Asking for details about a named movie
    - "What is [movie title]?", "Find the movie called..."

    Args:
        title: The movie title (or approximate title) the user is looking for.

    Returns:
        The answer from the fuzzy search agent.
    """
    result = fuzzy_ask(title)
    return result["answer"]


# ============================================================
# Agent construction
# ============================================================
def _build_router_agent():
    """Build the router agent."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        base_url="https://openai.vocareum.com/v1",
        api_key="voc-46640206319004520369916a7f3655c69c53.81266809",
    )

    tools = [sql_agent_tool, semantic_search_tool, fuzzy_title_search_tool]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=ROUTER_AGENT_PROMPT,
        checkpointer=InMemorySaver(),
    )
    return agent


# Lazy singleton
_router_agent = None


def get_router_agent():
    """Get or create the router agent singleton."""
    global _router_agent
    if _router_agent is None:
        _router_agent = _build_router_agent()
    return _router_agent


def ask(question: str, session_id: str | None = None) -> dict:
    """Route a question to the appropriate sub-agent and return the answer.

    Args:
        question: The user's question.
        session_id: Optional session ID for conversation continuity.

    Returns:
        dict with 'answer' and 'session_id'.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    agent = get_router_agent()
    config = {"configurable": {"thread_id": session_id}}

    result = agent.invoke({"messages": [("human", question)]}, config=config)
    answer = result["messages"][-1].content

    return {"answer": answer, "session_id": session_id}


def ask_stream(question: str, session_id: str | None = None):
    """Stream all router agent steps and final answer.

    Yields:
        dict with one of:
          - {"type": "tool_call", "tool": str, "input": str, "session_id": str}
          - {"type": "tool_result", "tool": str, "output": str, "session_id": str}
          - {"type": "token", "content": str, "session_id": str}
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    agent = get_router_agent()
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
