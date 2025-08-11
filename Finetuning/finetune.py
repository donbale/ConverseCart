import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# --- 1. Configuration ---
# The name of the base model to fine-tune. We use Llama 3.1 8B Instruct.
base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# The path to your JSONL dataset file.
dataset_path = "finetuning_dataset.jsonl"

# The name for your new, fine-tuned model on the Hugging Face Hub.
# Replace "YourUsername" with your actual Hugging Face username.
new_model_name = "YourUsername/ConverseCart-Llama3-8B-Instruct" 

# --- 2. Load the Dataset ---
# Load the dataset from the local JSONL file.
dataset = load_dataset("json", data_files=dataset_path, split="train")

# --- 3. Configure Model Quantization (for memory efficiency) ---
# This configuration loads the model in 4-bit precision, which drastically
# reduces memory usage and makes it possible to train on a single consumer GPU.
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
    device_map="auto", # Automatically select the device (GPU if available)
)
# The model's configuration should not use caching during training.
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load the tokenizer for the base model.
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
# The padding token should be set to the end-of-sentence token.
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# --- 5. Set Up Training Arguments ---
# These arguments define the parameters for the training process.
training_args = TrainingArguments(
    output_dir="./results",        # Directory to save training outputs.
    num_train_epochs=1,            # A single epoch is often enough for fine-tuning.
    per_device_train_batch_size=4, # Batch size per device during training.
    gradient_accumulation_steps=1, # Number of updates steps to accumulate before backprop.
    learning_rate=2e-4,            # The initial learning rate for the AdamW optimizer.
    weight_decay=0.001,            # Weight decay to apply.
    optim="paged_adamw_32bit",     # Use a memory-efficient optimizer.
    fp16=False,                    # Disable 16-bit precision training.
    bf16=True,                     # Enable bfloat16 for better performance on modern GPUs.
    max_grad_norm=0.3,             # Gradient clipping.
    max_steps=-1,                  # Number of training steps. -1 means train for num_train_epochs.
    warmup_ratio=0.03,             # Ratio of steps for a linear warmup.
    logging_steps=25,              # Log every 25 steps.
    push_to_hub=True,              # Push the final model to the Hugging Face Hub.
    hub_model_id=new_model_name,   # The repository name on the Hub.
)

# --- 6. Create the SFTTrainer and Start Training ---
# The SFTTrainer is designed for supervised fine-tuning.
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="prompt", # The name of the field in your dataset that contains the input text.
    # We need to format the dataset to be a chat-like format.
    # The 'completion' field is the expected response.
    formatting_func=lambda example: [{"role": "user", "content": example["prompt"][0]}, {"role": "assistant", "content": example["completion"][0]}],
    max_seq_length=512,          # Maximum sequence length.
    tokenizer=tokenizer,
    args=training_args,
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# --- 7. Push the model to the Hub ---
# This will save the final model adapter and push it to the Hub.
trainer.push_to_hub()
print(f"Model pushed to Hub at: https://huggingface.co/{new_model_name}")