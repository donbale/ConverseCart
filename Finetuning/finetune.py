import json, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training

# =============== Config ===============
DATASET_PATH   = "finetuning_dataset.jsonl"  # <-- use the 5k set we generated
BASE_MODEL_NAME = "google/gemma-2b-it"
NEW_MODEL_NAME  = "YOURUSERNAME/ConverseCart-Gemma-2B-Instruct"

# Keep the instruction aligned with resolver-friendly schema
SYSTEM_PROMPT = (
    "You control an e-commerce web app by returning ONE JSON object with keys "
    "\"action\" and \"payload\". Valid actions: \"setPage\", \"addToCart\", \"removeFromCart\".\n"
    "Payload rules:\n"
    "- setPage: {\"name\":\"productList\"|\"cart\"} OR {\"name\":\"productDetail\",\"productId\":\"...\"} "
    "  OR {\"name\":\"productDetail\",\"productRef\":{\"name\":\"...\"}}\n"
    "- addToCart: {\"productId\":\"...\"} OR {\"productRef\":{\"name\":\"...\"}}\n"
    "- removeFromCart: {\"productId\":\"...\"} OR {\"productRef\":{\"name\":\"...\"}}\n"
    "Do NOT add any text outside the JSON. Output JSON only."
)

# =============== Load dataset ===============
raw = load_dataset("json", data_files=DATASET_PATH, split="train")

def _must_be_json(ex):
    c = ex.get("completion","").strip()
    json.loads(c)  # assert valid JSON
    return {"prompt": ex.get("prompt","").strip(), "completion": c}

ds = raw.map(_must_be_json, remove_columns=[c for c in raw.column_names if c not in ("prompt","completion")])

def create_gemma_prompt(example):
    # Gemma chat template: <start_of_turn>system/user/model ... <end_of_turn>
    return (
        f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
        f"<start_of_turn>user\n{example['prompt']}<end_of_turn>\n"
        f"<start_of_turn>model\n{example['completion']}<end_of_turn>"
    )

formatted = ds.map(lambda ex: {"text": create_gemma_prompt(ex)}, remove_columns=ds.column_names)

# =============== 4-bit QLoRA ===============
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

print(f"Loading base model: {BASE_MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, quantization_config=bnb, device_map="auto")
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 512

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
)

sft_config = SFTConfig(
    output_dir="./results",
    num_train_epochs=2,                     # 2–3 is plenty here
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,          # eff batch 8/GPU
    optim="paged_adamw_32bit",
    learning_rate=5e-5,                     # safer than 2e-4 for JSON fidelity
    weight_decay=0.01,
    fp16=False, bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    logging_steps=25,
    push_to_hub=True,
    hub_model_id=NEW_MODEL_NAME,
    dataset_text_field="text",
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=formatted,
    peft_config=peft_config,
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

trainer.push_to_hub()
print(f"Model pushed to Hub at: https://huggingface.co/{NEW_MODEL_NAME}")
