def end_max_iterations(state):
    state["query_result"] = "Please try again."
    return state

def check_attempts_router(state):
    return "convert_to_sql" if state["attempts"] < 3 else "end_max_iterations"