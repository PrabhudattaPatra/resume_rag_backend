import os
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_redis import RedisSemanticCache
from langchain_core.globals import set_llm_cache
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

# Optional: Add Redis Semantic Caching with Logging
REDIS_URL = os.getenv("REDIS_URL")
if REDIS_URL:
    class LoggingRedisCache(RedisSemanticCache):
        def lookup(self, prompt: str, llm_string: str):
            result = super().lookup(prompt, llm_string)
            if result:
                print(f"🎯 [CACHE HIT] Serving response from Redis Semantic Cache for: '{prompt[:50]}...'")
            else:
                print(f"☁️ [CACHE MISS] Querying LLM for: '{prompt[:50]}...'")
            return result

    try:
        # Use Semantic Cache (fuzzy matching) instead of standard cache (exact match)
        semantic_cache = LoggingRedisCache(
            redis_url=REDIS_URL,
            embeddings=embeddings,
            ttl=600,
            distance_threshold=0.1  # Adjust similarity sensitivity
        )
        set_llm_cache(semantic_cache)
        print("🚀 Redis Semantic Cache enabled with logging.")
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}. Proceeding without cache.")

# If grader needs a different LLM or we want to abstract it:
response_model = llm
grader_model = llm

# Advanced model specifically for data ingestion and high-fidelity PDF extraction
ingestion_llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)
