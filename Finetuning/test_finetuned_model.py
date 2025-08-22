import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

BASE_MODEL_ID = "google/gemma-2b-it"
PEFT_MODEL_ID = "donbale/ConverseCart-Gemma-2B-Instruct"

SYSTEM_PROMPT = (
    "You control an e-commerce web app by returning ONE JSON object with keys "
    "\"action\" and \"payload\". Valid actions: \"setPage\", \"addToCart\", \"removeFromCart\".\n"
    "Payload rules:\n"
    "- setPage: {\"name\":\"productList\"|\"cart\"} OR {\"name\":\"productDetail\",\"productId\":\"...\"} "
    "  OR {\"name\":\"productDetail\",\"productRef\":{\"name\":\"...\"}}\n"
    "- addToCart: {\"productId\":\"...\"} OR {\"productRef\":{\"name\":\"...\"}}\n"
    "- removeFromCart: {\"productId\":\"...\"} OR {\"productRef\":{\"name\":\"...\"}}\n"
    "Output JSON only."
)

def make_prompt(user_text: str) -> str:
    return (
        f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def first_json_obj(s: str) -> str:
    i = s.find("{")
    if i == -1: return s
    depth = 0; end = None
    for j, ch in enumerate(s[i:], start=i):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    return s[i:end] if end else s

if __name__ == "__main__":
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, PEFT_MODEL_ID)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    def run(q):
        prompt = make_prompt(q)
        terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")]
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=terminators,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0], skip_special_tokens=False)
        # slice off the prompt
        gen_tail = gen[len(prompt):].strip()
        print("Raw:\n", gen_tail)
        try:
            print("Parsed:\n", json.dumps(json.loads(first_json_obj(gen_tail)), indent=2))
        except Exception as e:
            print("JSON parse error:", e)

    tests = [
        "Add the Quantum Laptop to my cart",
        "Tell me more about the noise cancelling headphones",
        "Show me all the products you have.",
        "What's in my basket?",
        "Open product prod_004",
        "remove prod_006 from cart",
    ]
    for t in tests:
        print("\nQ:", t)
        run(t)
        print("="*60)