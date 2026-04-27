import os
from langchain_pinecone import PineconeVectorStore
from langchain.tools import tool
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from rag.config import pc, embeddings, llm
from rag.ingestion.loaders import load_cgu_md, load_fee_structure_md

# Connect to Pinecone Indices
doc_index = pc.Index("doc-index")
exam_index = pc.Index("examination-index")
notice_index = pc.Index("cgu-notice-index")

# --- Initialize BM25 Retrievers ---
# Hybrid search requires the text content for the keyword search part (BM25)
# We load the documents locally on startup to build the BM25 index
try:
    # 1. Main Docs (FAQ + Fees)
    doc_pages = load_cgu_md("data/cgu.md") + load_fee_structure_md("data/fee_structure.md")
    doc_bm25 = BM25Retriever.from_documents(doc_pages)
    doc_bm25.k = 2

    # 2. Examination Docs (If you have a local copy, load it here)
    # For now, we'll use a placeholder or the same pages if specific files aren't available
    exam_bm25 = BM25Retriever.from_documents(doc_pages) # TODO: Load actual exam docs
    exam_bm25.k = 2

    # 3. Notice Docs
    notice_bm25 = BM25Retriever.from_documents(doc_pages) # TODO: Load actual notice docs
    notice_bm25.k = 2
except Exception as e:
    print(f"Warning: Could not load local documents for BM25: {e}")
    # Fallback: if data is missing, BM25 will be skipped in the ensemble
    doc_bm25 = exam_bm25 = notice_bm25 = None

# --- Create Vector Stores ---
doc_store = PineconeVectorStore(index=doc_index, embedding=embeddings)
exam_store = PineconeVectorStore(index=exam_index, embedding=embeddings)
notice_store = PineconeVectorStore(index=notice_index, embedding=embeddings)

# --- Create Base Vector Retrievers ---
# We increase 'k' here because the re-ranker needs more candidates to choose from
doc_retriever = doc_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5,"lambda_mult": 0.6}
)

exam_retriever = exam_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5,"lambda_mult": 0.5}
)

notice_retriever = notice_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5,"lambda_mult": 0.5}
)

# --- Create Ensemble Retrievers (Hybrid Search) ---
if doc_bm25:
    doc_ensemble = EnsembleRetriever(
        retrievers=[doc_bm25, doc_retriever],
        weights=[0.3, 0.7]
    )
    exam_ensemble = EnsembleRetriever(
        retrievers=[exam_bm25, exam_retriever],
        weights=[0.3, 0.7]
    )
    notice_ensemble = EnsembleRetriever(
        retrievers=[notice_bm25, notice_retriever],
        weights=[0.3, 0.7]
    )
else:
    doc_ensemble = doc_retriever
    exam_ensemble = exam_retriever
    notice_ensemble = notice_retriever

# --- Add Re-ranking Layer (Level 5) ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if COHERE_API_KEY:
    try:
        compressor = CohereRerank(model="rerank-english-v3.0", top_n=4)
        
        doc_ensemble_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=doc_ensemble
        )
        exam_ensemble_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=exam_ensemble
        )
        notice_ensemble_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=notice_ensemble
        )
        print("Cohere Re-ranker enabled.")
    except Exception as e:
        print(f"Failed to initialize Cohere Reranker: {e}")
        doc_ensemble_retriever, exam_ensemble_retriever, notice_ensemble_retriever = doc_ensemble, exam_ensemble, notice_ensemble
else:
    print("COHERE_API_KEY missing. Proceeding without Re-ranker.")
    doc_ensemble_retriever, exam_ensemble_retriever, notice_ensemble_retriever = doc_ensemble, exam_ensemble, notice_ensemble

# Tools
@tool
def retrieve_blog_posts(query: str) -> str:
    """
    Search and return official university information, including:
    - University overview, rankings, and accreditations
    - Admissions process and entrance exams
    - Eligibility criteria for all courses
    - Scholarships and education loans
    - Office roles, responsibilities, and contact details
    - Fees, payments, and refund policies
    - Campus policies, facilities, and student life
    - International student admissions and requirements
    - Placements, recruiters, and career outcomes
    """
    docs = doc_ensemble_retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


@tool
def retrieve_examination_cell_doc(query: str) -> str:
    """
    Search and return official university exam notifications,
    schedules, admit card releases, result notices, and 
    examination circulars.
    """
    docs = exam_ensemble_retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


@tool
def retrieve_notice_board_doc(query: str) -> str:
    """
    Search and return official university notices and announcements
    from the CGU Odisha Notice Board ,
    including:
    - Admit card distribution updates
    - Time table for any course/Branch
    - Notifications about academic activities
    - Convocation and award announcements
    - Other official circulars and events
    """
    docs = notice_ensemble_retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# Optional: Add TavilySearch or other tools if needed
