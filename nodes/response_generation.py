from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent.sql_agent import llm



def generate_human_readable_answer(state):
    system = """Bạn là một trợ lý AI giúp diễn giải kết quả truy vấn SQL thành câu trả lời rõ ràng và dễ hiểu bằng tiếng Việt. Lưu ý chỉ trả lời 1 lượt"""
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
Trả lời câu hỏi của người dùng **bằng tiếng Việt** và có thể thêm chút dí dỏm. Lưu ý chỉ trả lời 1 lượt"""
    
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


def generate_human_readable_answer_vinallama(state):
    sql = state["sql_query"]
    result = state.get("query_rows", [])
    
    if state.get("sql_error", False):
        user_instruction = f"""SQL Query:
{sql}
Result:
{result}
Hãy diễn giải lỗi xảy ra bằng tiếng Việt."""
    elif not result:
        user_instruction = f"""SQL Query:
{sql}
Result:
{result}
Không có dữ liệu nào được trả về. Hãy đưa ra một câu trả lời phù hợp bằng tiếng Việt."""
    else:
        user_instruction = f"""SQL Query:
{sql}
Result:
{result}
Hãy diễn giải kết quả truy vấn một cách rõ ràng bằng tiếng Việt."""

    chatml_prompt = f"""<|im_start|>system
Bạn là một trợ lý AI giúp diễn giải kết quả truy vấn SQL thành câu trả lời dễ hiểu bằng tiếng Việt.<|im_end|>
<|im_start|>user
{user_instruction}<|im_end|>
<|im_start|>assistant
"""

    response = llm.invoke(chatml_prompt)
    clean_text = extract_chatml_answer(response)

    state["query_result"] = clean_text
    return state

import re

def extract_chatml_answer(raw_output: str) -> str:
    if "<|im_start|>assistant" in raw_output:
        raw_output = raw_output.split("<|im_start|>assistant")[1].strip()

    if "<|im_end|>" in raw_output:
        raw_output = raw_output.split("<|im_end|>")[0].strip()

    return raw_output

def generate_funny_response_vinallama(state):
    human_message = state["question"]
    
    chatml_prompt = f"""<|im_start|>system
Bạn là một trợ lý AI thông minh, thân thiện và hài hước. Trả lời câu hỏi của người dùng bằng tiếng Việt và có thể thêm chút dí dỏm.<|im_end|>
<|im_start|>user
{human_message}<|im_end|>
<|im_start|>assistant
"""

    response = llm.invoke(chatml_prompt)
    clean_text = extract_chatml_answer(response)

    state["query_result"] = clean_text
    return state
