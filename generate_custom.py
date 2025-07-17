from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_dir = "/Users/sudharsanraja/Desktop/LLM/results/checkpoint-129"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
theme = "certification"
tone = "confident"
prompt = f"### Theme: {theme}\n### Tone: {tone}\n### Instruction: Write a {tone} LinkedIn post about {theme}.\n### Post:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_post = output_text.split("### Post:")[-1].strip()
print("\n Final LinkedIn Post:\n")
print(generated_post)