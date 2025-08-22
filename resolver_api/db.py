# resolver_api/db.py
import os
import re
import sqlite3
from typing import List, Dict, Optional

# --- Config (override via env or docker-compose) ---
DB_PATH = os.getenv("CONVERSECART_DB_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "conversecart.db")))
CATALOG_MD_PATH = os.getenv("CONVERSECART_CATALOG_MD", os.path.abspath(os.path.join(os.path.dirname(__file__), "product-catalog.md")))

# --- Schema (products + FTS5 mirror with sync triggers) ---
SCHEMA_SQL = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS products (
  product_id   TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  category     TEXT NOT NULL,
  price        REAL NOT NULL,
  description  TEXT NOT NULL,
  stock        INTEGER NOT NULL CHECK (stock >= 0),
  created_at   TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_products_name     ON products(name);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);

-- Full-text search across name/category/description
CREATE VIRTUAL TABLE IF NOT EXISTS products_fts
USING fts5(product_id, name, category, description, content='products', content_rowid='rowid');

-- Keep FTS in sync with base table
CREATE TRIGGER IF NOT EXISTS products_ai AFTER INSERT ON products BEGIN
  INSERT INTO products_fts(rowid, product_id, name, category, description)
  VALUES (new.rowid, new.product_id, new.name, new.category, new.description);
END;

CREATE TRIGGER IF NOT EXISTS products_ad AFTER DELETE ON products BEGIN
  INSERT INTO products_fts(products_fts, rowid, product_id, name, category, description)
  VALUES ('delete', old.rowid, old.product_id, old.name, old.category, old.description);
END;

CREATE TRIGGER IF NOT EXISTS products_au AFTER UPDATE ON products BEGIN
  INSERT INTO products_fts(products_fts, rowid, product_id, name, category, description)
  VALUES ('delete', old.rowid, old.product_id, old.name, old.category, old.description);
  INSERT INTO products_fts(rowid, product_id, name, category, description)
  VALUES (new.rowid, new.product_id, new.name, new.category, new.description);
END;
"""

# --- Markdown parser for product-catalog.md ---
def parse_catalog_markdown(md_text: str) -> List[Dict]:
    """
    Expects blocks like:

    ### Product ID: `prod_001`
    **Name**: Quantum Laptop
    **Category**: Electronics
    **Price**: $1299.99
    **Description**: ...
    **Stock**: 15
    """
    parts = re.split(r"###\s*Product ID:\s*`([^`]+)`", md_text)
    products: List[Dict] = []

    for pid, block in zip(parts[1::2], parts[2::2]):
        def grab(label: str, b: str = block) -> Optional[str]:
            m = re.search(rf"\*\*{re.escape(label)}\*\*:\s*(.+)", b)
            return m.group(1).strip() if m else None

        name = grab("Name")
        category = grab("Category")
        price_s = grab("Price") or ""
        description = grab("Description")
        stock_s = grab("Stock") or ""

        # Coerce types
        m_price = re.search(r"(\d+(?:\.\d+)?)", price_s)
        price = float(m_price.group(1)) if m_price else 0.0

        m_stock = re.search(r"(\d+)", stock_s)
        stock = int(m_stock.group(1)) if m_stock else 0

        if not (pid and name and category and description):
            continue

        products.append({
            "product_id": pid,
            "name": name,
            "category": category,
            "price": price,
            "description": description,
            "stock": stock,
        })

    return products

# --- SQLite helpers ---
def get_conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db(db_path: Optional[str] = None, catalog_md_path: Optional[str] = None) -> int:
    """
    Rebuild the DB from the markdown catalog.
    Returns the number of products inserted.
    """
    dbp = db_path or DB_PATH
    catp = catalog_md_path or CATALOG_MD_PATH

    with open(catp, "r", encoding="utf-8") as f:
        md_text = f.read()
    products = parse_catalog_markdown(md_text)

    # Recreate DB for a clean state
    if os.path.exists(dbp):
        os.remove(dbp)

    con = get_conn(dbp)
    cur = con.cursor()
    cur.executescript(SCHEMA_SQL)

    cur.executemany("""
        INSERT INTO products (product_id, name, category, price, description, stock)
        VALUES (:product_id, :name, :category, :price, :description, :stock)
        ON CONFLICT(product_id) DO UPDATE SET
          name=excluded.name,
          category=excluded.category,
          price=excluded.price,
          description=excluded.description,
          stock=excluded.stock,
          updated_at=datetime('now')
    """, products)
    con.commit()
    con.close()
    return len(products)

def search_products(con: sqlite3.Connection, q: str, limit: int = 5) -> List[Dict]:
    """
    Prefer FTS (MATCH); fallback to LIKE if FTS5 unavailable.
    """
    # Try FTS
    try:
        rows = con.execute(
            """
            SELECT p.product_id, p.name, p.category, p.price, p.description, p.stock
            FROM products_fts f
            JOIN products p ON p.rowid = f.rowid
            WHERE products_fts MATCH ?
            LIMIT ?
            """,
            (q, limit)
        ).fetchall()
        if rows:
            return [dict(r) for r in rows]
    except Exception:
        pass  # FTS might not be available; fallback below

    # Fallback: simple LIKE on name/category/description
    like = f"%{q}%"
    rows = con.execute(
        """
        SELECT product_id, name, category, price, description, stock
        FROM products
        WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
        LIMIT ?
        """,
        (like, like, like, limit)
    ).fetchall()
    return [dict(r) for r in rows]

def get_product(con: sqlite3.Connection, product_id: str) -> Optional[Dict]:
    row = con.execute(
        """
        SELECT product_id, name, category, price, description, stock
        FROM products
        WHERE product_id = ?
        """,
        (product_id,)
    ).fetchone()
    return dict(row) if row else None
