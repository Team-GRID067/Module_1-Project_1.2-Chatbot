from dotenv import load_dotenv
from graph.workflow import app
from agent.state import AgentState
load_dotenv()

# Example usage
if __name__ == "__main__":
    input_state = {
        "question": "Show me top 10 customers by sales",
        "sql_query": "",
        "query_result": "",
        "query_rows": [],
        "attempts": 0,
        "relevance": "",
        "sql_error": False
    }
    state= AgentState(**input_state)
    for output in app.stream(state):
        for key, value in output.items():
            print(f"Node '{key}':")
            print(value["question"], value["sql_query"], value["query_result"], sep="\n")