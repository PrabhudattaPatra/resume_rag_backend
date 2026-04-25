import pytest
import json
from fastapi.testclient import TestClient

# Mock the environment variables before importing the app to prevent real connections
import os
os.environ["DATABASE_URL"] = "postgresql://mock_user:mock_pass@localhost:5432/mock_db"
os.environ["PINECONE_API_KEY"] = "mock-pinecone-key"
os.environ["GOOGLE_API_KEY"] = "mock-google-key"
os.environ["GROQ_API_KEY"] = "mock-groq-key"
os.environ["TAVILY_API_KEY"] = "mock-tavily-key"

from app import app
import app as app_module

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint returns a 200 OK and expected message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "LangGraph Chatbot API is running" in response.json()["message"]

def test_home_endpoint():
    """Test the /home endpoint returns a 200 OK."""
    response = client.get("/home")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the C.V. Raman Global University RAG Chatbot API"}

def test_health_check_unhealthy(monkeypatch):
    """
    Test the health check when the database is disconnected.
    Since we provided a fake DB_URI, the lifespan will fail to connect
    and compiled_graph will be None.
    """
    # Ensure compiled_graph is None
    monkeypatch.setattr(app_module, "compiled_graph", None)
    
    response = client.get("/health")
    # Our app returns 200 even for unhealthy status, it just changes the JSON payload.
    assert response.status_code == 200
    assert response.json()["status"] == "unhealthy"
    assert "error" in response.json()

def test_health_check_healthy(monkeypatch):
    """
    Test the health check when the graph is successfully compiled.
    """
    # Mock the compiled graph to simulate a successful connection
    monkeypatch.setattr(app_module, "compiled_graph", "mock_compiled_graph")
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_endpoint_db_failure(monkeypatch):
    """
    Test that the /api/chat endpoint correctly returns a 503 Service Unavailable
    when the database connection is broken (compiled_graph is None).
    """
    monkeypatch.setattr(app_module, "compiled_graph", None)
    
    payload = {"message": "Hello!"}
    response = client.post("/api/chat", json=payload)
    
    assert response.status_code == 503
    assert "Chatbot engine is currently unavailable" in response.json()["detail"]

def test_chat_endpoint_validation_error():
    """
    Test that missing the required 'message' field results in a 422 Unprocessable Entity.
    """
    payload = {"thread_id": "12345"} # Missing 'message'
    response = client.post("/api/chat", json=payload)
    
    assert response.status_code == 422
