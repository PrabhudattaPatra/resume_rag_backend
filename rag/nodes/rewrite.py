from langchain_core.messages import HumanMessage, ToolMessage
from rag.state import AgentState, get_latest_user_question
from rag.config import response_model

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:\n"
    " ------- \n"
    "{question}\n"
    " ------- \n"
    "Formulate an improved question:"
)

def rewrite_question(state: AgentState):
    """Rewrite the original user question."""
    messages = state["messages"]
    
    # Safety: do not rewrite if tool already returned content
    if any(isinstance(m, ToolMessage) and m.content.strip() for m in messages):
        return {"rewrite_count": state.get("rewrite_count", 0)}
        
    question = get_latest_user_question(messages)
    current_count = state.get("rewrite_count", 0)
    
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    
    return {
        "messages": [HumanMessage(content=response.content)],
        "rewrite_count": current_count + 1
    }
