# resolver_api/main.py
import os
from typing import List

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from db import (
    get_conn,
    init_db,
    search_products as db_search_products,
    get_product as db_get_product,
    DB_PATH,
    CATALOG_MD_PATH,
)

app = FastAPI(title="ConverseCart Resolver API", version="0.1.0")

# --- CORS (adjust origins as needed) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # for local dev; lock down for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB dependency per request ---
def get_db():
    con = get_conn(DB_PATH)
    try:
        yield con
    finally:
        con.close()

# --- Response models ---
class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    description: str
    stock: int

# --- Routes ---
@app.get("/health")
def health(db = Depends(get_db)):
    row = db.execute("SELECT COUNT(*) AS c FROM products").fetchone()
    return {
        "status": "ok",
        "db_path": DB_PATH,
        "catalog_md": CATALOG_MD_PATH,
        "product_count": row["c"] if row else 0,
    }

@app.get("/search_products", response_model=List[Product])
def search_products(
    q: str = Query(..., min_length=1, description="Search text"),
    limit: int = Query(5, ge=1, le=50, description="Max results"),
    db = Depends(get_db),
):
    return db_search_products(db, q, limit)

@app.get("/get_product/{product_id}", response_model=Product)
def get_product(product_id: str, db = Depends(get_db)):
    prod = db_get_product(db, product_id)
    if not prod:
        raise HTTPException(status_code=404, detail="product not found")
    return prod

@app.post("/reload_catalog")
def reload_catalog():
    count = init_db(DB_PATH, CATALOG_MD_PATH)
    return {"status": "ok", "reloaded": count, "db_path": DB_PATH}

@app.get("/")
def root():
    return {"status": "ok", "message": "Resolver API is running."}

# --- Run standalone (optional) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
