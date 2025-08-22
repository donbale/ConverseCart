# resolver_api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (main.py, db.py, db_init.py, product-catalog.md, etc.)
COPY . .

# Default env (can be overridden in docker-compose)
ENV CONVERSECART_DB_PATH=/app/conversecart.db
ENV CONVERSECART_CATALOG_MD=/app/product-catalog.md

# Initialize the SQLite DB from product-catalog.md at build time
# (You can rebuild this later via POST /reload_catalog)
RUN python db_init.py --db $CONVERSECART_DB_PATH --catalog $CONVERSECART_CATALOG_MD || true

EXPOSE 8000

# Start the API
CMD ["sh", "-c", "python db_init.py --db $CONVERSECART_DB_PATH --catalog $CONVERSECART_CATALOG_MD || true && uvicorn main:app --host 0.0.0.0 --port 8000"]
