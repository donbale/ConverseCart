import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the functions from your existing RAG pipeline script
# Make sure the script is named 'main.py' and is in the same directory.
from main import build_rag_pipeline, get_action_from_query

# --- 1. Initialize FastAPI App and RAG Pipeline ---

# Create the FastAPI app instance
app = FastAPI(
    title="Conversational Commerce API",
    description="An API that uses a RAG pipeline to turn natural language into e-commerce actions.",
    version="1.0.0",
)

# Load the RAG pipeline once at startup
# This is crucial for performance, so we don't rebuild it on every request.
print("Server starting up...")
rag_pipeline = build_rag_pipeline()
print("RAG Pipeline loaded and ready.")


# --- 2. Configure CORS ---

# Configure Cross-Origin Resource Sharing (CORS) to allow your React app
# to communicate with this server.
# The local React app runs on http://localhost:8000
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)


# --- 3. Define API Request Models ---

# Pydantic model for the incoming request body.
# This ensures that the 'query' field is a string.
class QueryRequest(BaseModel):
    query: str


# --- 4. Define API Endpoints ---

@app.get("/", tags=["Status"])
def read_root():
    """A simple endpoint to check if the server is running."""
    return {"status": "ok", "message": "Conversational Commerce API is running."}


@app.post("/query", tags=["RAG Pipeline"])
def process_user_query(request: QueryRequest):
    """
    Accepts a user's natural language query and returns a JSON action.
    """
    if not rag_pipeline:
        return {"error": "RAG pipeline is not available. Please check server logs."}
    
    # Get the action from the RAG pipeline using the user's query
    action = get_action_from_query(rag_pipeline, request.query)
    
    return action


# --- 5. Run the Server ---

# This block allows you to run the server directly with `python server.py`
# Uvicorn is a lightning-fast ASGI server.
if __name__ == "__main__":
    print("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)