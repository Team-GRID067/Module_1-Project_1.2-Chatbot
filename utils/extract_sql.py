import re

def extract_sql(output: str) -> str:
    """
    Trích xuất câu truy vấn SQL từ đầu ra của LLM (có thể chứa markdown, văn bản phụ...).
    """
    match = re.search(r"```sql\s*(.*?)\s*```", output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    for line in output.splitlines():
        line = line.strip()
        if re.match(r"^(SELECT|WITH|INSERT|UPDATE|DELETE)", line, re.IGNORECASE):
            return line
    return output.strip()