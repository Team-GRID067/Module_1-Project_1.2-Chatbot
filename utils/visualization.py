import re
import matplotlib.pyplot as plt
import pandas as pd
from sql_db.engine import SessionLocal
from sqlalchemy import text, inspect

def drawing_chart(question, result):
    session = SessionLocal()
    return_result = session.execute(text(result["sql_query"]))
    all_results = return_result.fetchall()
    
    if not all_results:
        return
    
    labels, values = zip(*all_results)
    
    if len(all_results) > 10 or "top" in question.lower():
        top_n = 10
        if "top" in question.lower():
            match = re.search(r'\d+', question)
            top_n = int(match.group()) if match else 10
        
        df = pd.DataFrame(all_results, columns=["Label", "Value"])
        df = df.nlargest(top_n, "Value")
        plt.figure(figsize=(12, 6))
        plt.barh(df["Label"], df["Value"])
        plt.tight_layout()
    else:
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values)
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.show()