#  LinkedIn LLM Post Generator (100% Offline)

Generate human-like, professional LinkedIn posts locally using a fine-tuned open-source LLM — no APIs, no internet, just your machine.

![Model](https://img.shields.io/badge/Model-TinyLLaMA--1.1B--Chat-blue)
![Fine-Tuning](https://img.shields.io/badge/PEFT-LoRA-green)
![Platform](https://img.shields.io/badge/Offline-Support-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

##  Objective

Build a fully offline LinkedIn post generator using TinyLLaMA-1.1B-Chat, fine-tuned on a handcrafted dataset with themes like internships, certifications, promotions, and project completions.

---

## Summary

- Built using **TinyLLaMA-1.1B-Chat**, a lightweight yet capable open-source model.
- Fine-tuned locally on MacBook Air M3 using **LoRA (PEFT)**.
- Dataset of 340+ prompt-response pairs for tone and theme diversity.
- Custom Python code for inference, theme/tone control, and post generation.
- Output: Natural, tailored LinkedIn posts — **offline and open-source**.

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/linkedin-llm-post-generator.git
cd linkedin-llm-post-generator
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install the key libraries:

```bash
pip install transformers datasets peft bitsandbytes accelerate torch
```

> On Mac (Apple Silicon), make sure `torch` uses the MPS backend.

### 4. Run Fine-Tuning (Optional)

```bash
python finetune_tinyllama.py
```

This fine-tunes TinyLLaMA using `linkedin_posts_dataset.jsonl` and `expanded_linkedin_dataset.jsonl`.

### 5. Generate Custom Post

```bash
python generate_custom.py --theme internship --tone humble
```

Or modify the input JSON to:
```json
{ "theme": "certification", "tone": "confident" }
```

---

##  Dataset Format

Example (JSONL):
```json
{
  "instruction": "Write a humble LinkedIn post about internship",
  "theme": "internship",
  "tone": "humble",
  "output": "<Generated post text>"
}
```

---

## Project Structure

| File                     | Purpose |
|--------------------------|---------|
| `finetune_tinyllama.py`  | Fine-tune the model with PEFT (LoRA) |
| `generate_custom.py`     | Generate a post using the fine-tuned model |
| `save_model.py`          | Save intermediate model checkpoints |
| `save_final_model.py`    | Save the final model to disk |
| `test_phi2.py`           | Optional test with Phi-2 model |
| `linkedin_posts_dataset.jsonl` | Core training dataset |
| `expanded_linkedin_dataset.jsonl` | Expanded version for broader coverage |

---

##  Achievements

✅ Created a custom dataset tailored to LinkedIn communication  
✅ Successfully fine-tuned TinyLLaMA locally on Apple M3  
✅ Runs 100% offline — no OpenAI, no APIs  
✅ Generated posts for 20+ themes with multiple tones  
✅ Designed for low-resource hardware (8–16GB RAM)

---

## Tech Stack

- **Model:** TinyLLaMA-1.1B-Chat
- **Frameworks:** Hugging Face Transformers, PEFT, Datasets
- **Optimization:** LoRA, Bitsandbytes
- **Hardware:** MacOS with MPS backend (Apple Silicon)

---

## What You’ll Learn

- Prompt engineering & dataset curation
- PEFT (LoRA) fine-tuning on consumer hardware
- Tokenization & decoding workflows
- Ethical local AI system design
- End-to-end LLM lifecycle on-device

---


---

## 🪄 License

This project is licensed under the **MIT License**.

---
