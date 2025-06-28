from langchain_core.prompts import ChatPromptTemplate
from agent.state import CheckRelevance
from agent.sql_agent import llm
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
    check_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    structured_llm = llm.with_structured_output(CheckRelevance)
    relevance_checker = check_prompt | structured_llm
    relevance = relevance_checker.invoke({})
    state["relevance"] = relevance.relevance
    return state

def relevance_router(state):
    return "convert_to_sql" if state["relevance"].lower() == "relevant" else "generate_funny_response"