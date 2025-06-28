from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent.sql_agent import llm

def generate_human_readable_answer(state):
    system = """You are an assistant that converts SQL query results into clear, natural language responses."""
    sql = state["sql_query"]
    result = state.get("query_rows", [])
    
    if state.get("sql_error", False):
        prompt_template = """SQL Query:\n{sql}\nResult:\n{result}\nFormulate a clear error message."""
    elif not result:
        prompt_template = """SQL Query:\n{sql}\nResult:\n{result}\nFormulate a clear answer."""
    else:
        prompt_template = """SQL Query:\n{sql}\nResult:\n{result}\nFormulate a clear answer."""
    
    generate_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", prompt_template)])
    human_response = generate_prompt | llm | StrOutputParser()
    state["query_result"] = human_response.invoke({"sql": sql, "result": str(result)})
    return state

def generate_funny_response(state):
    system = """You are a charming and funny assistant who responds in a playful manner."""
    human_message = "I can not help with that, but doesn't asking questions make you come closer to the problem?"
    funny_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human_message)])
    funny_response = funny_prompt | llm | StrOutputParser()
    state["query_result"] = funny_response.invoke({})
    return state