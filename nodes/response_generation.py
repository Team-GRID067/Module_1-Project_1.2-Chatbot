from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent.sql_agent import llm



def generate_human_readable_answer(state):
    system = """Bạn là một trợ lý AI giúp diễn giải kết quả truy vấn SQL thành câu trả lời rõ ràng và dễ hiểu bằng tiếng Việt."""
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
    system = """Bạn là một trợ lý AI thông minh, thân thiện và hài hước. 
Trả lời câu hỏi của người dùng **bằng tiếng Việt** và có thể thêm chút dí dỏm."""
    
    human_message = state['question']
    
    funny_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{input}")
    ])
    
    funny_response = funny_prompt | llm | StrOutputParser()
    state["query_result"] = funny_response.invoke({"input": human_message})
    return state