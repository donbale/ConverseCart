import os
import re
import logging
import requests
from typing import Optional, Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm_local import LocalLLM

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("rag.server")

LLM_MODE = os.getenv("LLM_MODE", "LOCAL_HF")
RESOLVER_BASE = os.getenv("RESOLVER_BASE_URL", "http://resolver:8000")

app = FastAPI(title="Conversational Commerce API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

local_llm: Optional[LocalLLM] = None

class QueryRequest(BaseModel):
    query: str
    ui_context: Optional[dict] = None

def resolve_product_ref(ref: dict | None) -> Optional[Dict[str, Any]]:
    if not ref or not isinstance(ref, dict):
        return None
    try:
        if ref.get("productId"):
            r = requests.get(f"{RESOLVER_BASE}/get_product/{ref['productId']}", timeout=5)
            if r.status_code == 200:
                return r.json()
            return None
        if ref.get("name"):
            r = requests.get(f"{RESOLVER_BASE}/search_products", params={"q": ref["name"], "limit": 5}, timeout=5)
            if r.status_code != 200:
                return None
            hits = r.json() or []
            return hits[0] if hits else None
    except Exception as e:
        log.exception("resolve_product_ref error: %s", e)
        return None
    return None

STOPWORDS = {
    "do","you","have","any","a","an","the","show","me","please","pls",
    "some","of","your","my","i","we","is","are","there","this","it","that","these","they"
}
BROWSE_HINTS = {"show","list","browse","catalog","range","see","available","carry","stock","sell","have"}
PRONOUNS = {"this","it","that","these","they"}

def normalize_for_search(q: str) -> List[str]:
    s = (q or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in STOPWORDS]
    if not toks:
        return []
    terms: List[str] = []
    whole = " ".join(toks)
    if whole:
        terms.append(whole)
    last = toks[-1]
    terms.append(last)
    if last.endswith("s") and len(last) > 3:
        terms.append(last[:-1])
    seen: set[str] = set()
    out: List[str] = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

def is_browse_intent(q: str) -> bool:
    s = (q or "").lower()
    return any(w in s for w in BROWSE_HINTS)

def is_generic_browse(q: str) -> bool:
    s = (q or "").lower()
    return bool(re.search(r"\b(all|everything|catalog|products?)\b", s)) and not bool(
        re.search(r"\b(mug|monitor|laptop|keyboard|headphones?|ssd)\b", s)
    )

def contains_add_intent(q: str) -> bool:
    return bool(re.search(r"\b(add|put|pop)\b.*\b(cart|basket)\b", (q or "").lower()))

def contains_remove_intent(q: str) -> bool:
    return bool(re.search(r"\b(remove|delete|take\s*out)\b.*\b(cart|basket)\b", (q or "").lower()))

def is_pronoun_query(q: str) -> bool:
    toks = re.findall(r"[a-z]+", (q or "").lower())
    return any(t in PRONOUNS for t in toks)

def resolver_search_names_normalized(query: str, limit: int = 8) -> List[dict]:
    for term in normalize_for_search(query) or [query]:
        try:
            r = requests.get(f"{RESOLVER_BASE}/search_products", params={"q": term, "limit": limit}, timeout=5)
            if r.status_code == 200:
                hits = r.json() or []
                out = []
                for h in hits:
                    if not isinstance(h, dict):
                        continue
                    pid = h.get("product_id") or h.get("productId")
                    nm  = h.get("name")
                    if not nm:
                        continue
                    out.append({"productId": pid, "name": nm, "price": h.get("price")})
                if out:
                    return out
        except Exception as e:
            log.exception("resolver_search_names_normalized error for term=%r: %s", term, e)
    return []

def _normalize_product_id(pid: str) -> Optional[str]:
    if not isinstance(pid, str):
        return None
    pid = pid.strip()
    m = re.fullmatch(r"(?:prod[_\s-]?)?0*(\d{1,3})", pid, flags=re.IGNORECASE)
    if m:
        return f"prod_{int(m.group(1)):03d}"
    return pid if re.fullmatch(r"prod_\d{3}", pid) else None

def canonicalize_for_frontend(action: dict):
    if not isinstance(action, dict) or "action" not in action or "payload" not in action:
        return {"error": "invalid action", "raw": action}

    act = action["action"]
    payload = action.get("payload") or {}

    if act == "setPage":
        name = payload.get("name")
        if name in ("productList", "cart"):
            return {"action": "setPage", "payload": {"name": name}}
        if name == "productDetail":
            ref = payload.get("productRef") or ({"productId": payload["productId"]} if "productId" in payload else None)
            prod = resolve_product_ref(ref) if ref else None
            if prod and "product_id" in prod:
                return {"action": "setPage", "payload": {"name": "productDetail", "productId": prod["product_id"]}}
            if "productId" in payload:
                pid = _normalize_product_id(payload["productId"]) or payload["productId"]
                return {"action": "setPage", "payload": {"name": "productDetail", "productId": pid}}
            return {"action": "setPage", "payload": {"name": "productList"}}
        return {"action": "setPage", "payload": {"name": "productList"}}

    if act == "addToCart":
        ref = payload.get("productRef") or ({"productId": payload.get("productId")} if "productId" in payload else None)
        prod = resolve_product_ref(ref) if ref else None
        if prod and "product_id" in prod:
            return {"action": "addToCart", "payload": {"id": prod["product_id"]}}
        if "id" in payload:
            pid = _normalize_product_id(payload["id"]) or payload["id"]
            return {"action": "addToCart", "payload": {"id": pid}}
        if "productId" in payload:
            pid = _normalize_product_id(payload["productId"]) or payload["productId"]
            return {"action": "addToCart", "payload": {"id": pid}}
        return {"error": "could not resolve product for addToCart", "raw": action}

    if act == "removeFromCart":
        ref = payload.get("productRef") or ({"productId": payload.get("productId")} if "productId" in payload else None)
        prod = resolve_product_ref(ref) if ref else None
        if prod and "product_id" in prod:
            return {"action": "removeFromCart", "payload": {"id": prod["product_id"]}}
        if "id" in payload:
            pid = _normalize_product_id(payload["id"]) or payload["id"]
            return {"action": "removeFromCart", "payload": {"id": pid}}
        if "productId" in payload:
            pid = _normalize_product_id(payload["productId"]) or payload["productId"]
            return {"action": "removeFromCart", "payload": {"id": pid}}
        return {"error": "could not resolve product for removeFromCart", "raw": action}

    return {"error": "unknown action", "raw": action}

def maybe_force_product_list(front_action: dict, user_query: str) -> dict:
    try:
        if front_action.get("action") == "setPage":
            name = front_action.get("payload", {}).get("name")
            if name == "cart" and is_browse_intent(user_query):
                return {"action": "setPage", "payload": {"name": "productList"}}
            if name == "productDetail" and is_generic_browse(user_query):
                return {"action": "setPage", "payload": {"name": "productList"}}
    except Exception:
        pass
    return front_action

def maybe_resolve_pronoun(front_action: dict, user_query: str, ui_context: dict | None) -> dict:
    try:
        if not (ui_context and ui_context.get("name") == "productDetail" and ui_context.get("productId")):
            return front_action
        if not is_pronoun_query(user_query):
            return front_action
        pid_ctx = ui_context["productId"]
        if front_action.get("action") == "setPage" and front_action.get("payload", {}).get("name") == "productDetail":
            pid = front_action.get("payload", {}).get("productId")
            if pid and pid != pid_ctx:
                return {"action": "setPage", "payload": {"name": "productDetail", "productId": pid_ctx}}
            if not pid:
                return {"action": "setPage", "payload": {"name": "productDetail", "productId": pid_ctx}}
        if front_action.get("action") in ("addToCart", "removeFromCart"):
            payload = dict(front_action.get("payload") or {})
            if "id" not in payload and "productId" not in payload:
                return {"action": front_action["action"], "payload": {"id": pid_ctx}}
    except Exception:
        pass
    return front_action

@app.on_event("startup")
def on_startup():
    global local_llm
    log.info("Starting RAG server with LLM_MODE=%s, RESOLVER_BASE=%s", LLM_MODE, RESOLVER_BASE)
    local_llm = LocalLLM()
    log.info("Local LLM ready")

@app.get("/health")
def health():
    return {"status": "ok", "mode": LLM_MODE, "pipeline_ready": local_llm is not None}

@app.post("/query")
def query(req: QueryRequest):
    if not local_llm:
        return {"error": "local model not ready"}

    raw = local_llm.infer(req.query, ui_context=req.ui_context)
    log.info("[infer] user_query=%r ui_ctx=%s -> raw_action=%s", req.query, req.ui_context, raw)
    if isinstance(raw, dict) and raw.get("error"):
        return raw

    front = canonicalize_for_frontend(raw)
    front = maybe_force_product_list(front, req.query)
    front = maybe_resolve_pronoun(front, req.query, req.ui_context)

    if "error" in front:
        if req.ui_context and req.ui_context.get("name") == "productDetail" and req.ui_context.get("productId"):
            pid_ctx = req.ui_context["productId"]
            if raw.get("action") == "addToCart" or contains_add_intent(req.query):
                front = {"action": "addToCart", "payload": {"id": pid_ctx}}
            elif raw.get("action") == "removeFromCart" or contains_remove_intent(req.query):
                front = {"action": "removeFromCart", "payload": {"id": pid_ctx}}
        if "error" in front and is_generic_browse(req.query):
            front = {"action": "setPage", "payload": {"name": "productList"}}

    log.info("[canon] -> %s", front)
    if "error" in front:
        return {**front, "reply": "I couldn’t find that just now — want to browse the catalog instead?"}

    prod = None
    try:
        if front["action"] == "setPage" and front["payload"].get("name") == "productDetail":
            pid = front["payload"].get("productId")
            if pid:
                prod = resolve_product_ref({"productId": pid})
        elif front["action"] in ("addToCart", "removeFromCart"):
            pid = front["payload"].get("id")
            if pid:
                prod = resolve_product_ref({"productId": pid})
    except Exception as e:
        log.exception("resolve current product error: %s", e)
        prod = None

    if not prod and isinstance(req.ui_context, dict):
        ctx_pid = req.ui_context.get("productId")
        if ctx_pid:
            try:
                prod = resolve_product_ref({"productId": ctx_pid})
            except Exception:
                prod = None

    if is_pronoun_query(req.query) and prod and prod.get("name"):
        candidates = [{
            "productId": prod.get("product_id") or prod.get("productId"),
            "name": prod.get("name"),
            "price": prod.get("price"),
        }]
    else:
        candidates = resolver_search_names_normalized(req.query, limit=8)

    allowed_names = [prod["name"]] if (prod and prod.get("name")) else [c["name"] for c in candidates if c.get("name")]

    log.info(
        "[ground] candidates=%s allowed=%s selected=%s",
        [c.get("name") for c in candidates],
        allowed_names,
        (prod or {}).get("name"),
    )

    reply = local_llm.generate_reply(
        user_query=req.query,
        action=front["action"],
        payload=front.get("payload", {}),
        product=prod,
        ui_context=req.ui_context or {},
        allowed_names=allowed_names,
        candidates=candidates,
    )
    log.info("[reply] %s", reply)

    return {**front, "reply": reply}
