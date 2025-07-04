from langgraph.graph import StateGraph, END
from agent.state import AgentState
from nodes.relevance import check_relevance, relevance_router
from nodes.query_generation import convert_nl_to_sql, regenerate_query
from nodes.execution import execute_sql, execute_sql_router
from nodes.response_generation import generate_human_readable_answer, generate_funny_response , generate_funny_response_vinallama, generate_human_readable_answer_vinallama
from nodes.error_handling import end_max_iterations, check_attempts_router
from nodes.retrieve_context import retrieve_context
from nodes.sql_injection import check_sensitive_query
def create_workflow():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("check_sensitive_query", check_sensitive_query)
    workflow.add_node("convert_to_sql", convert_nl_to_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("generate_human_readable_answer", generate_human_readable_answer_vinallama)
    workflow.add_node("generate_funny_response", generate_funny_response_vinallama)
    workflow.add_node("regenerate_query", regenerate_query)
    workflow.add_node("end_max_iterations", end_max_iterations)
    
    # Add edges
    workflow.add_conditional_edges(
        "check_relevance",
        relevance_router,
        {"retrieve_context": "retrieve_context", "generate_funny_response": "generate_funny_response","convert_to_sql":"convert_to_sql"},
    )
    workflow.add_edge("retrieve_context", "convert_to_sql")
    workflow.add_edge("convert_to_sql", "check_sensitive_query")
    workflow.add_edge("check_sensitive_query", "execute_sql")

    workflow.add_conditional_edges(
        "execute_sql",
        execute_sql_router,
        {"generate_human_readable_answer": "generate_human_readable_answer", "regenerate_query": "regenerate_query"}
    )
    workflow.add_conditional_edges(
        "regenerate_query",
        check_attempts_router,
        {"convert_to_sql": "convert_to_sql", "end_max_iterations": "end_max_iterations"}
    )
    workflow.add_edge("generate_human_readable_answer", END)
    workflow.add_edge("generate_funny_response", END)
    workflow.add_edge("end_max_iterations", END)
    
    workflow.set_entry_point("check_relevance")
    return workflow.compile()

app = create_workflow()