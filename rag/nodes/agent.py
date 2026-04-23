from rag.state import AgentState
from rag.config import response_model
from rag.retrievers import (
    retrieve_blog_posts, 
    retrieve_examination_cell_doc, 
    retrieve_notice_board_doc
)

def generate_query_or_respond(state: AgentState):
    """
    Call the model to generate a response based on the current state. 
    Given the question, it will decide to retrieve using the tools or simply respond to the user.
    """
    tools = [
        retrieve_blog_posts,
        retrieve_examination_cell_doc,
        retrieve_notice_board_doc
    ]
    
    response = response_model.bind_tools(tools).invoke(state["messages"])
    
    return {"messages": [response]}
