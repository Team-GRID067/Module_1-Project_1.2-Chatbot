from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent.sql_agent import llm



def generate_human_readable_answer(state):
    system = """Bạn là một trợ lý AI giúp diễn giải kết quả truy vấn SQL thành câu trả lời rõ ràng và dễ hiểu bằng tiếng Việt."""
    sql = state["sql_query"]
    result = state.get("query_rows", [])
    
    if state.get("sql_error", False):
        prompt_template = """SQL Query:
                            {sql}
                            Result:
                            {result}
                            Formulate a clear error message in Vietnamese."""
    elif not result:
        prompt_template = """SQL Query:
                            {sql}
                            Result:
                            {result}
                            Formulate a clear message indicating no data was found."""
    else:
        prompt_template = """SQL Query:
                            {sql}
                            Result:
                            {result}
                            Formulate a clear answer in Vietnamese."""
                                
    full_prompt = f"""{system}

    {prompt_template}"""
    
    prompt = ChatPromptTemplate.from_template(full_prompt)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"sql": sql, "result": str(result)})
    state["query_result"] = response
    return state


def generate_funny_response(state):
    system_message = """Bạn là một trợ lý AI thông minh, thân thiện và hài hước. 
Trả lời câu hỏi của người dùng **bằng tiếng Việt** và có thể thêm chút dí dỏm."""
    
    human_message = state['question']
    
    funny_prompt = ChatPromptTemplate.from_template(
            """{system_message}

        Người dùng: {input}
        AI:"""
        )

    
    funny_response = funny_prompt | llm | StrOutputParser()
    response = funny_response.invoke({
        "input": human_message,
        "system_message": system_message
    })  

    state["query_result"] = response
    return state