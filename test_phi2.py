from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
     model_name,
     torch_dtype=torch.float16,
     device_map="auto"
)
prompt = (
      "### Instruction:\n"
      "Write a short LinkedIn post about completing an NLP certification. "
      "Tone: confident, humble, and friendly.\n\n"
      "### Input:\n"
      "I completed my certification in NLP and LLMs.\n\n"
      "### Response:\n"
) 
inputs = tokenizer(prompt, return_tensors="pt").to("mps")
outputs = model.generate(**inputs, max_new_tokens=80)
print("\nGenerated Post:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
