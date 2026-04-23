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

# Global compiled graph — set during lifespan startup
compiled_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager.
    Opens the async Postgres connection pool on startup,
    sets up the checkpoint tables, compiles the graph, 
    then cleanly closes the pool on shutdown.
    """
    global compiled_graph
    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            # Creates checkpoint tables if they don't already exist
            await checkpointer.setup()
            compiled_graph = workflow.compile(checkpointer=checkpointer)
            print("✅ Postgres checkpointer ready. Graph compiled successfully.")
            yield  # Server runs here
        # Connection pool is automatically closed after yield
        print("🔌 Postgres connection pool closed.")
    except Exception as e:
        print(f"❌ Failed to initialize database connection: {e}")
        # Proceed with yielding so FastAPI can start gracefully, we'll handle the None compiled_graph in endpoints.
        yield


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
        error_data = json.dumps({"text": f"\n\n[System Error: {str(e)}]"})
        yield f"data: {error_data}\n\n"

@app.get("/health")
async def health_check():
    try:
        if compiled_graph is None:
            return {"status": "unhealthy", "service": "LangGraph Chatbot API", "error": "Database connection failed"}
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
        if compiled_graph is None:
            raise HTTPException(status_code=503, detail="Chatbot engine is currently unavailable due to database connection failure.")
            
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
