import uuid
import json
import os
from typing import Optional
from pydantic import BaseModel
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Import the uncompiled workflow from our rag package
from rag.graph import workflow

load_dotenv()
DB_URI = os.getenv("DATABASE_URL")

# Global uncompiled workflow
# We will compile it per-request to prevent Neon DB "connection closed" timeouts.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager.
    We removed the global DB connection pool because Neon Serverless Postgres 
    aggressively drops idle connections, causing "the connection is closed" errors.
    """
    print("✅ FastAPI Server starting up.")
    yield
    print("🔌 FastAPI Server shutting down.")
app = FastAPI(title="LangGraph Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def stream_generator(message: str, thread_id: str):
    try:
        # Create a fresh database connection for each request
        # This completely solves the "the connection is closed" error caused by idle Neon timeouts
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            compiled_graph = workflow.compile(checkpointer=checkpointer)
            
            config = {"configurable": {"thread_id": thread_id}}
            inputs = {"messages": [("user", message)]}

            async for message_chunk, metadata in compiled_graph.astream(
                inputs, 
                config=config, 
                stream_mode="messages"
            ):
                node_name = metadata.get("langgraph_node")
                
                if node_name in ["generate_answer", "generate_query_or_respond"]:
                    is_tool_call = hasattr(message_chunk, "tool_call_chunks") and message_chunk.tool_call_chunks
                    
                    if not is_tool_call:
                        chunk_text = ""
                        if hasattr(message_chunk, "content_blocks"):
                            for block in message_chunk.content_blocks:
                                if block["type"] == "text" and block["text"]:
                                    chunk_text += block["text"]
                        elif hasattr(message_chunk, "text") and message_chunk.text:
                            chunk_text = message_chunk.text
                        
                        if chunk_text:
                            data = json.dumps({"text": chunk_text})
                            yield f"data: {data}\n\n"
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print the full error stack trace to the Docker logs!
        error_data = json.dumps({"text": f"\n\n[System Error: {str(e)}]"})
        yield f"data: {error_data}\n\n"

@app.get("/health")
async def health_check():
    try:
        return {"status": "healthy", "service": "LangGraph Chatbot API"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/home")
async def home_endpoint():
    try:
        return {"message": "Welcome to the C.V. Raman Global University RAG Chatbot API"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

@app.post("/api/chat")
async def chat_endpoint(request_data: ChatRequest):
    try:

        message = request_data.message
        thread_id = request_data.thread_id or str(uuid.uuid4())
        
        return StreamingResponse(
            stream_generator(message, thread_id),
            media_type="text/event-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "LangGraph Chatbot API is running. Please access the frontend at its dedicated URL."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
