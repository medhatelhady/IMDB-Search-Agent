"""Streamlit chat application for the Movie Agent.

Run with: streamlit run movie_agent/ui/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import uuid
import streamlit as st
from movie_agent.agent.router import ask_stream, ask


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Movie Agent",
    page_icon="🎬",
    layout="centered",
)

# ============================================================
# Session state initialization
# ============================================================
if "sessions" not in st.session_state:
    # sessions: dict mapping session_id -> {"name": str, "messages": list}
    st.session_state.sessions = {}

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None


def create_new_session() -> str:
    """Create a new chat session and return its ID."""
    session_id = str(uuid.uuid4())
    st.session_state.sessions[session_id] = {
        "name": f"Chat {len(st.session_state.sessions) + 1}",
        "messages": [],
    }
    st.session_state.active_session_id = session_id
    return session_id


def get_active_session():
    """Get the active session data, creating one if none exists."""
    if st.session_state.active_session_id is None or \
       st.session_state.active_session_id not in st.session_state.sessions:
        create_new_session()
    return st.session_state.sessions[st.session_state.active_session_id]


# ============================================================
# Sidebar: session management
# ============================================================
with st.sidebar:
    st.title("🎬 Movie Agent")
    st.markdown("---")

    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        create_new_session()
        st.rerun()

    st.markdown("### Chat Sessions")

    # List existing sessions
    for sid, session_data in st.session_state.sessions.items():
        is_active = sid == st.session_state.active_session_id
        label = session_data["name"]
        if is_active:
            label = f"💬 {label}"

        if st.button(label, key=f"session_{sid}", use_container_width=True):
            st.session_state.active_session_id = sid
            st.rerun()

    if st.session_state.sessions:
        st.markdown("---")
        if st.button("🗑️ Clear All Sessions", use_container_width=True):
            st.session_state.sessions = {}
            st.session_state.active_session_id = None
            st.rerun()


# ============================================================
# Main chat area
# ============================================================
session = get_active_session()
session_id = st.session_state.active_session_id

st.title("Ask me about movies!")
st.caption(f"Session: `{session_id[:8]}...`")

# Display chat history
for msg in session["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            if msg.get("steps"):
                with st.expander("🔍 Agent Steps", expanded=False):
                    for step in msg["steps"]:
                        if step["type"] == "tool_call":
                            st.markdown(f"**🛠️ Calling:** `{step['tool']}`")
                            st.code(step["input"], language="text")
                        elif step["type"] == "tool_result":
                            st.markdown(f"**📋 Result from** `{step['tool']}`:")
                            st.code(step["output"][:1000], language="text")
            st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about movies..."):
    # Add user message to history
    session["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Auto-name the session based on first message
    if len(session["messages"]) == 1:
        session["name"] = prompt[:40] + ("..." if len(prompt) > 40 else "")

    # Stream assistant response
    with st.chat_message("assistant"):
        steps_container = st.expander("🔍 Agent Steps", expanded=True)
        response_placeholder = st.empty()

        full_response = ""
        steps = []

        try:
            for event in ask_stream(prompt, session_id=session_id):
                if event["type"] == "tool_call":
                    steps.append(event)
                    with steps_container:
                        st.markdown(f"**🛠️ Calling:** `{event['tool']}`")
                        st.code(event["input"], language="text")

                elif event["type"] == "tool_result":
                    steps.append(event)
                    with steps_container:
                        st.markdown(f"**📋 Result from** `{event['tool']}`:")
                        st.code(event["output"][:1000], language="text")

                elif event["type"] == "token":
                    full_response += event["content"]
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        except Exception as e:
            # Fallback to non-streaming if streaming fails
            try:
                result = ask(prompt, session_id=session_id)
                full_response = result["answer"]
                response_placeholder.markdown(full_response)
            except Exception as fallback_error:
                full_response = f"Sorry, I encountered an error: {fallback_error}"
                response_placeholder.markdown(full_response)

    # Add assistant message to history
    session["messages"].append({
        "role": "assistant",
        "content": full_response,
        "steps": steps,
    })
