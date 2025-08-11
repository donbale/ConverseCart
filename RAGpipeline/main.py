import os
import json
from dotenv import load_dotenv # Import the dotenv library
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate # Import PromptTemplate

# --- 0. Load Environment Variables ---
# This will load the GROQ_API_KEY from your .env file
load_dotenv()

# --- 1. Configuration ---
# Make sure your markdown files are in a 'knowledge_base' directory
# or update these paths.
KNOWLEDGE_BASE_DIR = "knowledge_base"
DOC_PATHS = [
    os.path.join(KNOWLEDGE_BASE_DIR, "api-docs.md"),
    os.path.join(KNOWLEDGE_BASE_DIR, "product-catalog.md"),
    os.path.join(KNOWLEDGE_BASE_DIR, "user-guides.md"),
]
# This is the model used to create vector embeddings.
# It runs locally on your machine.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# This is the model from Groq we will use for inference.
# Llama3 8b is a great, fast choice for this task.
GROQ_MODEL_NAME = "llama3-8b-8192"


# --- 2. Build the RAG Pipeline ---

def build_rag_pipeline():
    """
    Loads documents, creates a vector store, and sets up the RAG chain.
    """
    print("Building RAG pipeline...")
    
    # Load the documents from the specified paths
    documents = []
    for doc_path in DOC_PATHS:
        try:
            loader = TextLoader(doc_path)
            documents.extend(loader.load())
            print(f"Loaded document: {doc_path}")
        except Exception as e:
            print(f"Error loading document {doc_path}: {e}")
            return None

    if not documents:
        print("No documents were loaded. Please check the file paths.")
        return None

    # Split the documents into smaller chunks for better retrieval
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(docs)} chunks.")

    # Create the embeddings model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Create the FAISS vector store (our searchable index)
    print("Creating FAISS vector store...")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    print("Vector store created successfully.")

    # Set up the LLM. This example uses Groq for fast inference.
    # The API key is now loaded automatically from the .env file.
    print(f"Initializing LLM: {GROQ_MODEL_NAME} via Groq...")
    try:
        llm = ChatGroq(
            temperature=0, 
            model_name=GROQ_MODEL_NAME,
        )
        # A quick check to ensure the API key was loaded and is valid
        llm.invoke("test") 
    except ImportError:
        print("\nError: The 'langchain-groq' package is not installed.")
        print("Please install it with: pip install langchain-groq")
        return None
    except Exception as e:
        # This will catch errors if the API key is missing or invalid
        print(f"\nError initializing Groq LLM: {e}")
        print("Please make sure your GROQ_API_KEY is set correctly in the .env file.")
        return None
    
    # Define the prompt template string
    prompt_template_string = """
    SYSTEM INSTRUCTION: You are an expert AI assistant that controls an e-commerce web application by generating precise JSON commands.
    Your ONLY task is to convert the user's request into a single, specific JSON object representing a function call.
    Use the provided context to find the correct function names and product IDs.
    The JSON object must have two keys: "action" (the function name as a string) and "payload" (an object with the function's parameters).
    If the action requires no parameters, the payload should be an empty object {{}}.
    Do not add any explanations, apologies, or extra text. ONLY output the JSON command.

    CONTEXT:
    {context}

    USER QUERY:
    {question}

    ASSISTANT'S RESPONSE (JSON only):
    """
    
    # Create the PromptTemplate object
    prompt = PromptTemplate(
        template=prompt_template_string, input_variables=["context", "question"]
    )

    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt} # Pass the PromptTemplate object here
    )
    print("RAG pipeline built successfully.")
    return qa_chain


# --- 3. Define the Query Function ---

def get_action_from_query(qa_chain, query):
    """
    Takes a user query, runs it through the RAG chain, and returns the action.
    """
    if not qa_chain:
        return {"error": "RAG chain is not available."}
        
    print(f"\nProcessing query: '{query}'")
    
    # Run the query through the chain
    result = qa_chain({"query": query})
    
    # The raw output from the LLM
    llm_output = result['result'].strip()
    print(f"LLM Raw Output:\n---\n{llm_output}\n---")

    # Try to parse the LLM's output as JSON
    try:
        # Clean the output in case the LLM wraps it in markdown
        if llm_output.startswith("```json"):
            llm_output = llm_output.replace("```json\n", "").replace("\n```", "")
        
        action_json = json.loads(llm_output)
        return action_json
    except json.JSONDecodeError:
        print("Error: LLM output was not valid JSON.")
        return {"error": "Failed to parse LLM output.", "raw_output": llm_output}


# --- 4. Main Execution and Testing ---

if __name__ == "__main__":
    # First, create the knowledge base directory if it doesn't exist
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Creating directory '{KNOWLEDGE_BASE_DIR}'.")
        print("Please add 'api-docs.md', 'product-catalog.md', and 'user-guides.md' to this directory.")
        os.makedirs(KNOWLEDGE_BASE_DIR)
        # Exit if the directory was just created, as the files won't be there yet.
        exit()

    rag_pipeline = build_rag_pipeline()

    if rag_pipeline:
        # --- Test Cases ---
        test_queries = [
            "Show me all the products you have.",
            "Tell me more about the noise cancelling headphones",
            "I'd like to buy the smart mug",
            "What's in my shopping cart?",
            "add the quantum laptop to my basket"
        ]

        for q in test_queries:
            action = get_action_from_query(rag_pipeline, q)
            print(f"Action for '{q}':\n{json.dumps(action, indent=2)}\n" + "="*40)

