import torch
from langchain_openai import ChatOpenAI
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# from langchain.llms import HuggingFacePipeline
import os

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5
)

# def load_model(): 
#     model_name = "vilm/vinallama-7b-chat"

#     reconfig_model = BitsAndBytesConfig(
#         load_in_4bit=True, 
#         bnb_4bit_use_double_quant=True, 
#         bnb_4bit_compute_dtype=torch.bfloat16, 
#         bnb_4bit_quant_type="nf4"
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config = reconfig_model, 
#         device_map="auto"
#     )

#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     model_pipeline = pipeline(
#         "text-generation", 
#         model=model, 
#         tokenizer=tokenizer,
#         max_new_token=512, 
#         pad_token_id=tokenizer.eos_token_id, 
#         device_map="auto" 
#     )
#     return HuggingFacePipeline(pipeline=model_pipeline)

# llm=load_model()