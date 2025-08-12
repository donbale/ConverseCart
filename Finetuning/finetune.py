import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    # TrainingArguments,  # (originally used; replaced by SFTConfig below)
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training

# --- 1. Configuration ---
# The name of the base model to fine-tune.
base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# The path to your JSONL dataset file.
dataset_path = "finetuning_dataset.jsonl"

# The name for your new, fine-tuned model on the Hugging Face Hub.
# Replace "YourUsername" with your actual Hugging Face username.
new_model_name = "YourUsername/ConverseCart-Llama3.1-8B-Instruct"

# --- 2. Load the Dataset ---
# Load the dataset from the local JSONL file.
dataset = load_dataset("json", data_files=dataset_path, split="train")

# --- 3. Configure Model Quantization (for memory efficiency) ---
# This configuration loads the model in 4-bit precision.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# --- 4. Load the Base Model and Tokenizer ---
print(f"Loading base model: {base_model_name}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
# Enforce the effective sequence length since your TRL build doesn't accept max_seq_length:
tokenizer.model_max_length = 512

# --- 5. Pre-process the Dataset into a plain list of strings ---
# This approach is the most robust to library version changes.
def create_prompt(example):
    # Create a formatted string for each example.
    return f"### User:\n{example['prompt']}\n\n### Assistant:\n{example['completion']}"

# Use the .map() method to apply the formatting and create a new list.
formatted_dataset = dataset.map(lambda example: {'text': create_prompt(example)})

# --- QLoRA quick fix (attach trainable adapters for 4-bit finetuning) ---
# Prepare the quantized model for k-bit training and add a LoRA config.
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# --- 6. Set Up Training Arguments ---
# These arguments define the parameters for the training process.
# (Using SFTConfig — the TRL-native replacement for TrainingArguments in SFT flows.)
sft_config = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    logging_steps=25,
    push_to_hub=True,
    hub_model_id=new_model_name,

    # SFT-specific bits:
    dataset_text_field="text",
    packing=False,  # keep single samples intact
)

# --- 7. Create the SFTTrainer and Start Training ---
# The SFTTrainer is initialized with the most basic arguments to avoid conflicts.
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=formatted_dataset,   # Pass the pre-formatted dataset
    processing_class=tokenizer,        # Your TRL build expects this instead of tokenizer=
    peft_config=peft_config,           # Attach LoRA to enable finetuning on quantized model
    # no max_seq_length arg here; tokenizer.model_max_length is used
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# --- 8. Push the model to the Hub ---
# This will save the final model adapter and push it to the Hub.
trainer.push_to_hub()
print(f"Model pushed to Hub at: https://huggingface.co/{new_model_name}")