"""
FastAPI backend for LLM Council (LOCAL NTU VERSION - LLM-ONLY PERSISTENT MODELS)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
import json

from . import storage

# Council pipeline
from .council import (
    stage1_collect_responses,
    stage3_synthesize_final,
)

# Model warm start (optional but recommended)
from .model_loader import warm_start


app = FastAPI(title="LLM Council API (Persistent Mode)")


# =========================================================
# CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# STARTUP (MODEL LOADING)
# =========================================================
@app.on_event("startup")
async def startup_event():
    print("🚀 Initializing LLM Council backend...")
    warm_start()
    print("✅ All models loaded and cached")


# =========================================================
# REQUEST MODELS
# =========================================================
class SendMessageRequest(BaseModel):
    content: str


class Conversation(BaseModel):
    id: str
    created_at: str
    title: str
    messages: list


# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "LLM Council API (Persistent Mode)"
    }


# =========================================================
# CONVERSATIONS
# =========================================================
@app.get("/api/conversations")
async def list_conversations():
    return storage.list_conversations()


@app.post("/api/conversations")
async def create_conversation():
    conversation_id = str(uuid.uuid4())
    return storage.create_conversation(conversation_id)


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    convo = storage.get_conversation(conversation_id)
    if convo is None:
        raise HTTPException(status_code=404, detail="Not found")
    return convo


# =========================================================
# MAIN PIPELINE (NON-STREAMING)
# =========================================================
@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):

    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    storage.add_user_message(conversation_id, request.content)

    # ----------------------------
    # STAGE 1
    # ----------------------------
    stage1_results = await stage1_collect_responses(
        user_query=request.content
    )

    if not stage1_results:
        return {
            "stage1": [],
            "stage3": {
                "model": "error",
                "response": "All models failed to respond"
            }
        }

    # ----------------------------
    # STAGE 3
    # ----------------------------
    stage3_result = await stage3_synthesize_final(
        request.content,
        stage1_results
    )

    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        [],
        stage3_result
    )

    return {
        "stage1": stage1_results,
        "stage3": stage3_result
    }


# =========================================================
# STREAMING VERSION (SSE)
# =========================================================
@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):

    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    async def event_generator():

        try:
            storage.add_user_message(conversation_id, request.content)

            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"

            stage1_results = await stage1_collect_responses(
                user_query=request.content
            )

            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"

            stage3_result = await stage3_synthesize_final(
                request.content,
                stage1_results
            )

            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                [],
                stage3_result
            )

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)