# rag_server/Dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV LLM_MODE=LOCAL_HF
ENV BASE_MODEL_ID=google/gemma-2b-it
ENV PEFT_MODEL_ID=donbale/ConverseCart-Gemma-2B-Instruct
ENV RESOLVER_BASE_URL=http://resolver:8000

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application (server.py, main.py, knowledge_base/, .env example, etc.)
COPY . .

# Default resolver URL (Docker Compose overrides if needed)


EXPOSE 8001

# Start the FastAPI app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
