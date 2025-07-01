import re

def clean_ans(text): 
    MAX_LINE = 3
    if not text: 
        return ""
    splitted_text = re.split(r"Assistant:\s*", text, flags=re.IGNORECASE)
    content = splitted_text[-1] if len(splitted_text) > 1 else text
    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    cleaned = []
    for line in lines:
        if not cleaned or line != cleaned[-1]:
            cleaned.append(line)

    return "\n".join(cleaned[:max_lines])