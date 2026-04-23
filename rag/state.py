from langgraph.graph import MessagesState
from typing import Annotated
from langchain.messages import HumanMessage

class AgentState(MessagesState):
    rewrite_count: Annotated[int, "Number of question rewrites"] = 0
    answer_retry_count: Annotated[int, "Number of answer regenerations"] = 0

def get_latest_user_question(messages):
    """Helper to extract the last human message from the state."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""
