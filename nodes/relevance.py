from langchain_core.prompts import ChatPromptTemplate
from agent.state import CheckRelevance
from agent.sql_agent import llm, llm_gemini
from sql_db.schema import get_database_schema
from colorama import Fore, Style

def check_relevance(state, config):
    question = state["question"]
    schema = get_database_schema()
    system = f"""You are an assistant that determines whether a given question is related to the following database schema.
    
    Schema:
    {schema}
    
    Respond with only "relevant" or "not_relevant".
    """
    human = f"Question: {question}"
    prompt = f"{system}\n\n{human}"

    response = llm_gemini.invoke(prompt)

    
    relevance_str = response.lower() if isinstance(response, str) else str(response).lower()

    if "relevant" in relevance_str and "not" not in relevance_str:
        relevance = "relevant"
    else:
        relevance = "not_relevant"

    state["relevance"] = relevance
    return state

def relevance_router(state):
    return "convert_to_sql" if state["relevance"].lower() == "relevant" else "generate_funny_response"