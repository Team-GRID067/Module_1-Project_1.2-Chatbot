from sqlalchemy import text
from sql_db.engine import SessionLocal

def execute_sql(state):
    sql_query = state["sql_query"].strip()
    session = SessionLocal()
    try:
        result = session.execute(text(sql_query))
        if sql_query.lower().startswith("select"):
            rows = result.fetchall()
            columns = result.keys()
            state["query_rows"] = [dict(zip(columns, row)) for row in rows] if rows else []
            state["query_result"] = f"{len(state['query_rows'])} rows found." if rows else "No results found"
        state["sql_error"] = False
    except Exception as e:
        state["query_result"] = f"Error executing SQL query: {str(e)}"
        state["sql_error"] = True
    finally:
        session.close()
    return state

def execute_sql_router(state):
    return "generate_human_readable_answer" if not state.get("sql_error", False) else "regenerate_query"