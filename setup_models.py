from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import os

print("Downloading Embedding Model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model downloaded successfully!")

print("\nDownloading SLM (IBM Granite 400M)...")
slm_name = "ibm-granite/granite-3.0-1b-a400m-instruct"
slm_tokenizer = AutoTokenizer.from_pretrained(slm_name)
slm_model = AutoModelForCausalLM.from_pretrained(slm_name, trust_remote_code=True)
print("SLM downloaded successfully!")

print("\nDownloading LLM (Phi-3 Mini 3.8B)...")
llm_name = "microsoft/Phi-3-mini-4k-instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code=True)
print("LLM downloaded successfully!")

print("\nAll models downloaded and cached!")
print("You can now run: python -m streamlit run app.py")