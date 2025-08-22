import os
import json
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

KNOWLEDGE_BASE_DIR = "knowledge_base"
DOC_PATHS = [
    os.path.join(KNOWLEDGE_BASE_DIR, "api-docs.md"),
    os.path.join(KNOWLEDGE_BASE_DIR, "product-catalog.md"),
    os.path.join(KNOWLEDGE_BASE_DIR, "user-guides.md"),
]
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama3-8b-8192"


def _load_documents():
    docs = []
    for path in DOC_PATHS:
        loader = TextLoader(path)
        docs.extend(loader.load())
    return docs


def _build_vectorstore(docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs


def build_rag_pipeline():
    documents = _load_documents()
    vectorstore = _build_vectorstore(documents)
    retriever = vectorstore.as_retriever()

    llm = ChatGroq(temperature=0, model_name=GROQ_MODEL_NAME)
    llm.invoke("ping")

    prompt_text = """
SYSTEM INSTRUCTION:
You control an e-commerce web app by returning ONE JSON object describing the action to take.
Absolutely DO NOT include any extra text, explanations, or markdown fences—ONLY the JSON.

OUTPUT SCHEMA:
{
  "action": "setPage" | "addToCart" | "removeFromCart",
  "payload": { ... },           // see PAYLOAD RULES
  "reply": "Short, friendly sentence describing what you did."
}

PAYLOAD RULES:
- setPage:
    {"name":"productList"} |
    {"name":"cart"} |
    {"name":"productDetail","productId":"..."} |
    {"name":"productDetail","productRef":{"name":"..."}}
- addToCart:
    {"productId":"..."} OR {"productRef":{"name":"..."}}
- removeFromCart:
    {"productId":"..."} OR {"productRef":{"name":"..."}}

STYLE FOR reply:
- Warm, concise (≈1 sentence, <= 12 words).
- Describe the action specifically.
- Mention the product name when appropriate (e.g., “Added AI-Powered Smart Mug to your cart.”)
- Do not include JSON, quotes, code fences, or extra commentary.

CONTEXT:
{context}

USER QUERY:
{question}

ASSISTANT (JSON only, no backticks):
"""
    prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


def _extract_first_json_object(text: str) -> str:
    s = text.find("{")
    if s == -1:
        return text
    depth = 0
    end = None
    for i, ch in enumerate(text[s:], start=s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return text[s:end] if end else text


def get_action_from_query(qa_chain, query: str) -> Dict[str, Any]:
    if not qa_chain:
        return {"error": "RAG chain is not available."}

    result = qa_chain({"query": query})
    raw = str(result["result"]).strip()

    cleaned = _strip_code_fences(raw)
    json_candidate = _extract_first_json_object(cleaned)

    try:
        return json.loads(json_candidate)
    except Exception:
        return {"error": "Failed to parse LLM output.", "raw_output": raw}
