import re
import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from agent.state import AgentState

class SqlInjectionChecker:
    def __init__(self, 
                 model_name="cssupport/mobilebert-sql-injection-detect", 
                 tokenizer_name = "google/mobilebert-uncased"):    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer =  MobileBertTokenizer.from_pretrained(tokenizer_name)
        self.model = MobileBertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, text):
        inputs = self.tokenizer(text, padding=False, truncation=True, return_tensors='pt', max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class, probabilities[0][predicted_class].item()

sql_checker = SqlInjectionChecker()

def validate_sql(text,
                 sensitive_query = ["delete", "drop", "truncate", "alter", "update", "insert"]
                 ):
    
    query_lower = text.lower()
    for keyword in sensitive_query:
        # Dùng regex để tránh false positive (vd: "dropped" không match "drop")
        if re.search(rf"\b{keyword}\b", query_lower):
            return True

    predicted_class, confidence = sql_checker.predict(text)
    if predicted_class > 0.7:
        return True
    
    return False


def check_sensitive_query(state: AgentState):
    sql_query = state.get("sql_query", "")
    if validate_sql(sql_query):
        state["query_result"] = "🚫 The generated SQL query contains forbidden operations (like DELETE, DROP, etc) or SQL Injection."
        state["sql_error"] = True
    return state
