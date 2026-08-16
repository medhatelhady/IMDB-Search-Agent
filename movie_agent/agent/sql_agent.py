"""SQL Agent: natural language -> SQL query -> answer about movies database.

Uses LangGraph with PostgresSaver checkpointer for session-based memory.
"""

import os
import uuid
import warnings

from sqlalchemy import create_engine
from sqlalchemy.exc import SAWarning

# Suppress the pgvector 'vector' type warning from SQLAlchemy reflection
warnings.filterwarnings("ignore", message="Did not recognize type 'vector'", category=SAWarning)

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import (
    QuerySQLDatabaseTool,
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
)
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from movie_agent.config import DATABASE_URL
from movie_agent.agent.prompts import SQL_AGENT_PROMPT


def _build_agent():
    """Build and return the SQL agent."""
    engine = create_engine(DATABASE_URL)
    db = SQLDatabase(engine)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        base_url = "https://openai.vocareum.com/v1",
        api_key = "voc-46640206319004520369916a7f3655c69c53.81266809"
    )

    tools = [
        QuerySQLDatabaseTool(db=db),
        InfoSQLDatabaseTool(db=db),
        ListSQLDatabaseTool(db=db),
    ]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SQL_AGENT_PROMPT,
        checkpointer=InMemorySaver(),
    )
    return agent


# Lazy singleton so the agent is only created once
_agent = None


def get_agent():
    """Get or create the SQL agent singleton."""
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def ask(question: str, session_id: str | None = None) -> dict:
    """Ask the SQL agent a question with session-based memory.

    Args:
        question: The natural language question.
        session_id: Optional session/thread ID for conversation continuity.
                    If None, a new session is created.

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
    """Stream all agent steps: tool calls, tool results, and final answer tokens.

    Args:
        question: The natural language question.
        session_id: Optional session/thread ID for conversation continuity.

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
                # Agent node emitting tool calls
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        yield {
                            "type": "tool_call",
                            "tool": tool_call["name"],
                            "input": str(tool_call.get("args", {})),
                            "session_id": session_id,
                        }

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
