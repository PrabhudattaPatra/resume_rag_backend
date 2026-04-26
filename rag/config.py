import os
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize core clients and models
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define embedding model used across all indices
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Initialize the main language models
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# If grader needs a different LLM or we want to abstract it:
response_model = llm
grader_model = llm

# Advanced model specifically for data ingestion and high-fidelity PDF extraction
ingestion_llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)
