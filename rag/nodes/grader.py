from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage
from rag.state import AgentState, get_latest_user_question
from rag.config import grader_model

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

class GradeDocuments(BaseModel):  
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

def grade_documents(state: AgentState) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = get_latest_user_question(state["messages"])
    context = state["messages"][-1].content
    rewrite_count = state.get("rewrite_count", 0)

    prompt = GRADE_PROMPT.format(question=question, context=context)
    
    response = (
        grader_model
        .with_structured_output(GradeDocuments)
        .invoke([{"role": "user", "content": prompt}])
    )
    
    score = response.binary_score
    
    # Check if we've hit the rewrite limit
    if rewrite_count >= 2:
        return "generate_answer"
    
    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
