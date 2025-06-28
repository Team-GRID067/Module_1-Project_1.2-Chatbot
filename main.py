from dotenv import load_dotenv
from graph.workflow import app

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
    
    for output in app.stream(input_state):
        for key, value in output.items():
            print(f"Node '{key}':")
            print(value["question"], value["sql_query"], value["query_result"], sep="\n")