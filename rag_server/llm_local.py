import os
import json
import re
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login as hf_login

log = logging.getLogger("rag.llm")

# =========================
# Prompts
# =========================

SYSTEM_PROMPT = (
    'You control an e-commerce web app by returning ONE JSON object with keys '
    '"action" and "payload". Valid actions: "setPage", "addToCart", "removeFromCart".\n'
    'Payload rules:\n'
    '- setPage: {"name":"productList"|"cart"} OR {"name":"productDetail","productId":"..."} '
    '  OR {"name":"productDetail","productRef":{"name":"..."}}\n'
    '- addToCart: {"productId":"..."} OR {"productRef":{"name":"..."}}\n'
    '- removeFromCart: {"productId":"..."} OR {"productRef":{"name":"..."}}\n'
    'Do NOT add any text outside the JSON. Output JSON only.'
)

REPLY_SYSTEM = (
    "You are AI-Store, a helpful shopping assistant. "
    "Reply with ONE concise sentence (max ~28 words), no emojis. "
    "Ground your answer ONLY in the provided context. "
    "If the user asks a factual attribute (e.g., size), answer directly from the product description; "
    "if that attribute is missing, say it's not listed and offer a next step. "
    "Do not imply actions that didn’t happen; only mention removal if action=removeFromCart. "
    "You may ONLY mention product names that appear under CandidateProducts or the SelectedProduct name; "
    "if CandidateProducts is (none), do not mention any product names."
)

REPLY_SYSTEM_STRICT = (
    REPLY_SYSTEM
    + " Be precise and literal. Do not invent or guess. "
      "If action=setPage/productDetail and product name is provided, include that exact name."
)

# =========================
# Config
# =========================
BASE_MODEL_ID  = os.getenv("BASE_MODEL_ID", "google/gemma-2b-it")
PEFT_MODEL_ID  = os.getenv("PEFT_MODEL_ID", "donbale/ConverseCart-Gemma-2B-Instruct-V2")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "192"))

# =========================
# Helpers
# =========================

def _first_json_object(s: str) -> str:
    i = s.find("{")
    if i == -1:
        return s
    depth = 0
    end = None
    for j, ch in enumerate(s[i:], start=i):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    return s[i:end] if end else s

def _first_sentence(text: str, max_chars: int = 160) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    for stop in [". ", "! ", "? "]:
        i = s.find(stop)
        if 0 <= i <= max_chars:
            return s[: i + 1]
    return (s[:max_chars].rstrip() + ("…" if len(s) > max_chars else "")) if s else ""

def _build_prompt_action(user_text: str, ui_ctx: dict | None = None) -> str:
    # Keep UI context hint tiny and inside the user turn
    hint = ""
    if isinstance(ui_ctx, dict) and ui_ctx:
        name = ui_ctx.get("name") or ""
        pid  = ui_ctx.get("productId") or ""
        if name or pid:
            hint = "\n[ctx: page={}, productId={}]".format(name, pid)

    parts = [
        "<start_of_turn>system\n",
        SYSTEM_PROMPT,
        "<end_of_turn>\n",
        "<start_of_turn>user\n",
        user_text.strip(),
        hint,
        "<end_of_turn>\n",
        "<start_of_turn>model\n",
    ]
    return "".join(parts)

def _grounding_block(p: dict) -> str:
    name = (p.get("name") or "").strip()
    price = p.get("price")
    desc = _first_sentence(p.get("description") or "")
    lines = []
    if name:
        lines.append(f"name: {name}")
    if price is not None:
        try:
            lines.append(f"price: ${float(price):.2f}")
        except Exception:
            pass
    if desc:
        lines.append(f"description: {desc}")
    return "\n".join(lines) if lines else "(none)"

def _build_prompt_reply(ctx: dict, strict: bool = False) -> str:
    system = REPLY_SYSTEM_STRICT if strict else REPLY_SYSTEM

    user_query = (ctx.get("user_query") or "").strip()
    action     = ctx.get("action") or ""
    payload    = ctx.get("payload") or {}
    product    = ctx.get("product") or {}
    ui_ctx     = ctx.get("ui_context") or {}
    candidates = ctx.get("candidates") or []

    payload_json = json.dumps(payload, ensure_ascii=False)
    ui_json      = json.dumps(ui_ctx, ensure_ascii=False)
    selected_blk = _grounding_block(product)

    cand_lines = []
    for c in candidates:
        nm = c.get("name")
        pid = c.get("productId")
        price = c.get("price")
        if not nm:
            continue
        line = f"- {pid}: {nm}"
        if price is not None:
            try:
                line += f" — ${float(price):.2f}"
            except Exception:
                pass
        cand_lines.append(line)
    candidates_blk = "\n".join(cand_lines) if cand_lines else "(none)"

    parts = [
        "<start_of_turn>system\n", system, "<end_of_turn>\n",
        "<start_of_turn>user\n",
        "User query: ", user_query, "\n",
        "Action: ", action, "\n",
        "Payload: ", payload_json, "\n",
        "UI context: ", ui_json, "\n",
        "SelectedProduct:\n", selected_blk, "\n",
        "CandidateProducts:\n", candidates_blk, "\n",
        "Write ONE sentence only.\n",
        "<end_of_turn>\n",
        "<start_of_turn>model\n",
    ]
    return "".join(parts)

def _sanitize_reply(line: str) -> str:
    line = (line or "").splitlines()[0].strip(" `\"")
    # strip emojis/symbols
    line = "".join(ch for ch in line if ch.isprintable() and not (0x1F300 <= ord(ch) <= 0x1FAFF))
    return line

def _mentions_disallowed_name(reply: str, allowed: list[str], action: str) -> bool:
    if not reply:
        return False
    rlow = reply.lower()
    if any(n and n.lower() in rlow for n in allowed):
        return False
    if action == "setPage":
        if re.search(r"\b([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*)\b\s+(Keyboard|Headphones?|Monitor|Mug|SSD|Laptop)s?\b", reply):
            return True
    if re.search(r"\b\d{1,3}\s?(?:\"|″|in|inch|inches|cm|mm)\b.*\b(Keyboard|Headphones?|Monitor|Mug|SSD|Laptop)s?\b", reply, re.I):
        return True
    return False

def _mentions_improper_removal(reply: str, action: str) -> bool:
    if action == "removeFromCart":
        return False
    return bool(re.search(r"\b(remove|removed|delet(?:e|ed)|take(?:n)?\s+off|unavailable|recommend(?:s|ed)? removing|should remove)\b", reply, re.I))

def _enforce_action_wording(action: str, reply: str) -> bool:
    r = (reply or "").lower()
    if action == "addToCart":
        if "added" in r and not re.search(r"\bask(?:ed)?\b", r):
            return True
        return False
    if action == "removeFromCart":
        return "removed" in r
    return True

def _needs_retry(action: str, payload: dict, product: dict, reply: str, allowed_names: list[str]) -> bool:
    r = (reply or "").lower()
    if "ground your answer" in r or "write one sentence" in r:
        return True
    if _mentions_improper_removal(reply, action):
        return True
    if action == "setPage" and (payload or {}).get("name") == "productDetail":
        nm = (product.get("name") or "").strip().lower()
        if nm and nm not in r:
            return True
    if _mentions_disallowed_name(reply, allowed_names, action):
        return True
    if not _enforce_action_wording(action, reply):
        return True
    return False

def _fallback_reply(action: str, payload: dict, product: dict, candidates: list[dict] | None) -> str:
    name = (product.get("name") or "").strip()
    price = product.get("price")
    desc = _first_sentence(product.get("description") or "")
    if action == "setPage":
        page = (payload or {}).get("name")
        if page == "productList":
            return "Showing products—say a product name and I’ll open it."
        if page == "cart":
            return "Here’s your cart—ready to checkout or keep browsing?"
        if page == "productDetail":
            bits = []
            if name: bits.append(f"Opening {name}.")
            if desc: bits.append(desc)
            if price is not None:
                try: bits.append(f"Price: ${float(price):.2f}.")
                except Exception: pass
            bits.append("Ask about specs or say “add to cart”.")
            return " ".join(bits)
        return "Done—what should we do next?"
    if action == "addToCart":
        return f"Added {name or 'the item'} to your cart; view cart or keep browsing?"
    if action == "removeFromCart":
        return f"Removed {name or 'the item'} from your cart; need anything else?"
    return "Done—what should we do next?"

# =========================
# LLM Wrapper
# =========================

class LocalLLM:
    def __init__(self):
        token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        if token:
            hf_login(token=token, add_to_git_credential=False)

        print(f"[LocalLLM] Loading base on CPU: {BASE_MODEL_ID}")
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=token,
        )
        print(f"[LocalLLM] Attaching adapter: {PEFT_MODEL_ID}")
        self.model = PeftModel.from_pretrained(self.model, PEFT_MODEL_ID, token=token)

        print("[LocalLLM] Loading tokenizer…")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID, trust_remote_code=True, token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    @torch.no_grad()
    def infer(self, query: str, ui_context: dict | None = None):
        prompt = _build_prompt_action(query, ui_ctx=ui_context)
        eot_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        terminators = [self.tokenizer.eos_token_id] if eot_id is None else [self.tokenizer.eos_token_id, eot_id]
        inputs = self.tokenizer(prompt, return_tensors="pt")
        out = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = out[0, inputs["input_ids"].shape[-1]:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        try:
            return json.loads(_first_json_object(gen_text))
        except Exception:
            return {"error": "Failed to parse LLM output.", "raw_output": gen_text}

    @torch.no_grad()
    def generate_reply(
        self,
        user_query: str,
        action: str,
        payload: dict,
        product: dict | None,
        ui_context: dict | None = None,
        allowed_names: list[str] | None = None,
        candidates: list[dict] | None = None,
    ) -> str:
        allowed_names = allowed_names or []
        candidates = candidates or []
        ctx = {
            "user_query": user_query,
            "action": action,
            "payload": payload or {},
            "product": product or {},
            "ui_context": ui_context or {},
            "candidates": candidates,
        }

        prompt = _build_prompt_reply(ctx, strict=False)
        eot_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        terminators = [self.tokenizer.eos_token_id] if eot_id is None else [self.tokenizer.eos_token_id, eot_id]
        inputs = self.tokenizer(prompt, return_tensors="pt")
        out = self.model.generate(
            **inputs,
            max_new_tokens=72,
            do_sample=False,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        reply = _sanitize_reply(text)

        if _needs_retry(action, payload or {}, product or {}, reply, allowed_names):
            log.warning("[LLM Reply] Retry strict due to guardrail: reply='%s'", reply)
            prompt2 = _build_prompt_reply(ctx, strict=True)
            inputs2 = self.tokenizer(prompt2, return_tensors="pt")
            out2 = self.model.generate(
                **inputs2,
                max_new_tokens=72,
                do_sample=False,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            text2 = self.tokenizer.decode(out2[0, inputs2["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            reply2 = _sanitize_reply(text2)

            if _needs_retry(action, payload or {}, product or {}, reply2, allowed_names):
                log.error("[LLM Reply] Fallback after strict retry; reply2='%s'", reply2)
                return _fallback_reply(action, payload or {}, product or {}, candidates)

            return reply2

        return reply or _fallback_reply(action, payload or {}, product or {}, candidates)

def init_local_llm():
    return LocalLLM()
