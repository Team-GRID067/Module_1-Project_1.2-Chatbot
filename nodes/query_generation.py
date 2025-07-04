from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent.state import ConvertToSQL, RewrittenQuestion
from agent.sql_agent import llm_gemini 
from sql_db.schema import get_database_schema
from colorama import Fore, Style
from utils.extract_sql import extract_sql

def convert_nl_to_sql(state, config):
    question = state["question"]
    schema = get_database_schema()
    context = state.get("retrieved_context", "")

    system_prompt = f"""Bạn là một trợ lý AI giúp chuyển đổi câu hỏi bằng ngôn ngữ tự nhiên thành truy vấn SQL dựa trên cấu trúc cơ sở dữ liệu sau:

{schema}

Nếu có ngữ cảnh bổ sung, hãy tận dụng để làm rõ mục đích truy vấn:
{context}

Câu hỏi hiện tại là: '{{input}}'

Hãy tạo truy vấn SQL phù hợp, không giải thích gì thêm. Chỉ trả về câu SQL."""

    convert_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    sql_generator = convert_prompt | llm_gemini | StrOutputParser()

    try:
        raw_output = sql_generator.invoke({"input": question})
        sql = extract_sql(raw_output)  
        state["sql_query"] = sql
        state["sql_error"] = False
    except Exception as e:
        print(Fore.RED + f"[LỖI] Không thể tạo SQL: {e}" + Style.RESET_ALL)
        state["sql_query"] = ""
        state["sql_error"] = True

    return state


def regenerate_query(state):
    question = state["question"]

    system_prompt = """Bạn là một trợ lý AI chuyên viết lại câu hỏi để dễ chuyển đổi thành truy vấn SQL.
Giữ nguyên ý nghĩa ban đầu, nhưng làm cho rõ ràng và cụ thể hơn."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Câu gốc: {input}\nHãy viết lại câu này cho rõ ràng hơn để dễ tạo câu SQL.")
    ])

    try:
        rewriter = prompt | llm_gemini | StrOutputParser()
        rewritten = rewriter.invoke({"input": question})
        state["question"] = rewritten.strip()
        state["attempts"] += 1
    except Exception as e:
        print(Fore.RED + f"[LỖI] Không thể viết lại câu hỏi: {e}" + Style.RESET_ALL)
        state["sql_error"] = True

    return state