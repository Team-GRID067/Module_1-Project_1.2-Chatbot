from langchain_core.prompts import ChatPromptTemplate
from agent.state import ConvertToSQL, RewrittenQuestion
from agent.sql_agent import llm
from sql_db.schema import get_database_schema
from colorama import Fore, Style

def convert_nl_to_sql(state, config):
    question = state["question"]
    schema = get_database_schema()
    context = state.get("retrieved_context", "")
    system = f"""You are an assistant that converts natural language questions into SQL queries based on the following 
    schema:
    {schema}
    
    
    Additional Context:
    {context}
    
    The current question is '{question}'. Ensure that all query-related data is scoped to this question.
    
    Provide only the SQL query without any explanations. Alias columns appropriately to match the expected keys in the result.
    """
    convert_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "Question: {question}")]
    )
    structured_llm = llm.with_structured_output(ConvertToSQL)
    sql_generator = convert_prompt | structured_llm
    result = sql_generator.invoke({"question": question})
    state["sql_query"] = result.sql_query
    return state

def regenerate_query(state):
    question = state["question"]
    system = """You are an assistant that reformulates an original question to enable more precise SQL queries."""
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", f"Original Question: {question}\nReformulate...")]
    )
    structured_llm = llm.with_structured_output(RewrittenQuestion)
    rewriter = rewrite_prompt | structured_llm
    rewritten = rewriter.invoke({})
    state["question"] = rewritten.question
    state["attempts"] += 1
    return state