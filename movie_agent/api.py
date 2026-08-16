"""FastAPI application exposing the SQL agent as an API.

Run with: uvicorn movie_agent.api:app --reload --port 8000
"""

import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from movie_agent.agent.router import ask, ask_stream

app = FastAPI(
    title="Movie Agent API",
    description="Ask natural language questions about the movies database.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    session_id: str


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """Ask a question (non-streaming). Returns the full answer at once."""
    try:
        result = ask(request.question, session_id=request.session_id)
        return AnswerResponse(
            question=request.question,
            answer=result["answer"],
            session_id=result["session_id"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
def ask_question_stream(request: QuestionRequest):
    """Ask a question with streaming response (Server-Sent Events).

    Each SSE event contains a JSON object with one of:
      - {"type": "tool_call", "tool": "...", "input": "...", "session_id": "..."}
      - {"type": "tool_result", "tool": "...", "output": "...", "session_id": "..."}
      - {"type": "token", "content": "...", "session_id": "..."}
      - {"type": "done", "session_id": "..."}
    """
    def event_generator():
        try:
            session_id = None
            for chunk in ask_stream(request.question, session_id=request.session_id):
                session_id = chunk["session_id"]
                data = json.dumps(chunk)
                yield f"data: {data}\n\n"
            # Send final done event
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
        except Exception as e:
            error_data = json.dumps({"type": "error", "detail": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
