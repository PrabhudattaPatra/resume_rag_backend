from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from rag.state import AgentState
from rag.nodes.agent import generate_query_or_respond
from rag.nodes.grader import grade_documents
from rag.nodes.generation import generate_answer
from rag.nodes.rewrite import rewrite_question
from rag.retrievers import (
    retrieve_blog_posts, 
    retrieve_examination_cell_doc, 
    retrieve_notice_board_doc
)

def build_graph():
    workflow = StateGraph(AgentState)

    # Define the tools exactly as they are mapped in the agent
    tools_list = [
        retrieve_blog_posts,
        retrieve_examination_cell_doc,
        retrieve_notice_board_doc
    ]

    # Add nodes
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode(tools_list))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    
    # Define edges
    workflow.add_edge(START, "generate_query_or_respond")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `retrieve` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Return the uncompiled workflow.
    # The checkpointer will be bound at app startup (see app.py lifespan).
    return workflow

# Expose the uncompiled workflow for the FastAPI app to compile with a checkpointer
workflow = build_graph()
