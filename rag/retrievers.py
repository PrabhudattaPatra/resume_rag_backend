from langchain_pinecone import PineconeVectorStore
from langchain.tools import tool

from rag.config import pc, embeddings

# Connect to Pinecone Indices
# We assume these indices have been created by `init_db.py`
doc_index = pc.Index("doc-index")
exam_index = pc.Index("examination-index")
notice_index = pc.Index("cgu-notice-index")

# Create Vector Stores
doc_store = PineconeVectorStore(index=doc_index, embedding=embeddings)
exam_store = PineconeVectorStore(index=exam_index, embedding=embeddings)
notice_store = PineconeVectorStore(index=notice_index, embedding=embeddings)

# Create Retrievers
doc_retriever = doc_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.6}
)

exam_retriever = exam_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
)

notice_retriever = notice_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
)

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
    docs = doc_retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


@tool
def retrieve_examination_cell_doc(query: str) -> str:
    """
    Search and return official university exam notifications,
    schedules, admit card releases, result notices, and 
    examination circulars.
    """
    docs = exam_retriever.invoke(query)
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
    docs = notice_retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# Optional: Add TavilySearch or other tools if needed
