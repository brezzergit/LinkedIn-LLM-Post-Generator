from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
checkpoint_path = "./tinyllama_linkedin/checkpoint-6"
save_path = "./tinyllama_linkedin_final"
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Final model saved to:", save_path)