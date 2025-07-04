import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_google_genai import ChatGoogleGenerativeAI

import google.generativeai as genai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult, Generation
from langchain_core.runnables import RunnableConfig

def load_local_model():
    model_name = "vilm/vinallama-7b-chat"

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True 
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",  
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    return HuggingFacePipeline(pipeline=generation_pipeline)

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=GOOGLE_API_KEY)
# gemini_model = genai.GenerativeModel(model_name="gemini-pro")


# class GeminiChat(BaseChatModel):
#     """LangChain-compatible Gemini Chat wrapper"""

#     def _convert_messages_to_prompt(self, messages):
#         parts = []
#         for msg in messages:
#             if isinstance(msg, SystemMessage):
#                 parts.append(f"[System]: {msg.content}")
#             elif isinstance(msg, HumanMessage):
#                 parts.append(f"[User]: {msg.content}")
#             elif isinstance(msg, AIMessage):
#                 parts.append(f"[AI]: {msg.content}")
#         return "\n".join(parts)

#     def _call(self, messages, stop=None):
#         prompt = self._convert_messages_to_prompt(messages)
#         response = gemini_model.generate_content(prompt)
#         return response.text

#     def _generate(self, messages, stop=None, run_manager: RunnableConfig = None) -> LLMResult:
#         text = self._call(messages, stop)
#         ai_msg = AIMessage(content=text)
#         gen = ChatGeneration(message=ai_msg, text=text)
#         return LLMResult(generations=[[gen]])

#     @property
#     def _llm_type(self) -> str:
#         return "gemini-chat"


llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


llm = load_local_model()
