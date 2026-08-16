"""Reusable Streamlit UI components for the Movie Agent chat."""

import streamlit as st


def render_message(role: str, content: str):
    """Render a single chat message."""
    with st.chat_message(role):
        st.markdown(content)


def render_chat_history(messages: list):
    """Render full chat history."""
    for msg in messages:
        render_message(msg["role"], msg["content"])


def session_selector(sessions: dict, active_id: str | None) -> str | None:
    """Render a session selector in the sidebar and return selected session ID."""
    if not sessions:
        st.info("No active sessions. Start a new chat!")
        return None

    session_names = {sid: data["name"] for sid, data in sessions.items()}
    options = list(session_names.keys())
    labels = [session_names[sid] for sid in options]

    current_index = options.index(active_id) if active_id in options else 0

    selected_label = st.selectbox(
        "Select session",
        labels,
        index=current_index,
    )

    selected_id = options[labels.index(selected_label)]
    return selected_id
