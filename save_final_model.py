from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint_path = "./tinyllama_linkedin/checkpoint-6"
save_path = "./tinyllama_linkedin_final"
print(" Loading checkpoint")
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(" Saving final model")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to: {save_path}")
