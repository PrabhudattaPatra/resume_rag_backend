from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from rag.state import AgentState, get_latest_user_question
from rag.config import response_model

def generate_answer(state: AgentState):
    """Generate an answer using retrieved context."""
    question = get_latest_user_question(state["messages"])
    
    context_list = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
    context = "\n\n".join(context_list)
    retry_count = state.get("answer_retry_count", 0)

    GENERATE_PROMPT = [
        SystemMessage(content="You are an assistant for question-answering tasks for C.V Raman global university,Bhubaneswar,Odisha,India (CGU)."),
        HumanMessage(content=f"""
Use the following pieces of retrieved context to answer the question. 
If tools answer is not related to the user question, just say that you don't know. 
Question: {question}
Context: {context}
""")
    ]
    
    response = response_model.invoke(GENERATE_PROMPT)
    
    return {
        "messages": [response],
        "answer_retry_count": retry_count + 1
    }
