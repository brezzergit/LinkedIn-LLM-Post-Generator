from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch

dataset = load_dataset("json", data_files="/Users/sudharsanraja/Desktop/LLM/data/linkedIn_post_dataset.jsonl", split="train")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
def tokenize_fn(example):
    prompt = f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
tokenized_ds = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=False,
    bf16=False,
    report_to="none",
    remove_unused_columns=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)
trainer.train()