from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import Dict
import os
import gc


class LLMHandler:
    def __init__(self):
        self.device = "cpu"
        self.slm_model_name = "ibm-granite/granite-3.0-1b-a400m-instruct"
        self.llm_model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.slm_model = None
        self.slm_tokenizer = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.current_model = None
    
    def unload_models(self):
        if self.slm_model is not None:
            del self.slm_model
            self.slm_model = None
        if self.llm_model is not None:
            del self.llm_model
            self.llm_model = None
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def load_slm(self):
        if self.current_model == "llm":
            self.unload_models()
        
        if self.slm_model is None:
            local_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'slm')
            if os.path.isdir(local_path) and os.listdir(local_path):
                self.slm_tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
                self.slm_model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                self.slm_tokenizer = AutoTokenizer.from_pretrained(self.slm_model_name, trust_remote_code=True)
                self.slm_model = AutoModelForCausalLM.from_pretrained(
                    self.slm_model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            self.slm_model.eval()
            self.current_model = "slm"
    
    def load_llm(self):
        if self.current_model == "slm":
            self.unload_models()
        
        if self.llm_model is None:
            local_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'llm')
            if os.path.isdir(local_path) and os.listdir(local_path):
                self.llm_tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, trust_remote_code=True)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            self.llm_model.eval()
            self.current_model = "llm"
    
    def format_prompt(self, question: str, context: str) -> str:
        prompt = f"""You are a helpful assistant. Answer the question accurately and completely using ONLY the information from the given context. Be specific and provide details from the context.

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the context above
- Include relevant details and specifics from the context
- If the information is not in the context, say "Information not available in the documents"

Answer:"""
        return prompt
    
    def generate_with_slm(self, question: str, context: str) -> Dict:
        self.load_slm()
        prompt = self.format_prompt(question, context[:2000])
        
        try:
            inputs = self.slm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.slm_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=self.slm_tokenizer.eos_token_id
                )
            
            response = self.slm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response[len(prompt):].strip()
            
            return {
                'answer': answer,
                'model_used': 'SLM (IBM Granite 400M)',
                'model_name': self.slm_model_name,
                'success': True
            }
        except Exception as e:
            return {
                'answer': f"Error: {str(e)}",
                'model_used': 'SLM (IBM Granite 400M)',
                'model_name': self.slm_model_name,
                'success': False
            }
    
    def generate_with_llm(self, question: str, context: str) -> Dict:
        self.load_llm()
        prompt = self.format_prompt(question, context[:3000])
        
        try:
            inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )
            
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response[len(prompt):].strip()
            
            return {
                'answer': answer,
                'model_used': 'LLM (Phi-3 Mini 3.8B)',
                'model_name': self.llm_model_name,
                'success': True
            }
        except Exception as e:
            return {
                'answer': f"Error: {str(e)}",
                'model_used': 'LLM (Phi-3 Mini 3.8B)',
                'model_name': self.llm_model_name,
                'success': False
            }
    
    def generate(self, question: str, context: str, use_llm: bool = False) -> Dict:
        if use_llm:
            return self.generate_with_llm(question, context)
        else:
            return self.generate_with_slm(question, context)
