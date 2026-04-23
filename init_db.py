import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

def initialize_databases():
    load_dotenv()
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY environment variable not found.")
        return
        
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    indexes_to_create = [
        "doc-index",
        "examination-index",
        "cgu-notice-index"
    ]
    
    for index_name in indexes_to_create:
        if not pc.has_index(index_name):
            print(f"Creating Pinecone index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists.")
            
if __name__ == "__main__":
    initialize_databases()
