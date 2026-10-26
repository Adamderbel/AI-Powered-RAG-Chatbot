from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_ollama import ChatOllama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import os
import shutil
import json
import uuid
from typing import List, Optional, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configurations ---
FAISS_INDEX_PATH = "faiss_index"
CHAT_HISTORY_FILE = "chat_history.json"
CONTEXT_JSON_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "context_data.json")

# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    

class ResetResponse(BaseModel):
    message: str

class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, str]]

# --- Initialize FastAPI ---
app = FastAPI(title="JSON Knowledge Base QA API", version="1.0.0")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables ---
llm = ChatOllama(model="phi3:mini")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None

# --- Load Existing Data ---
def load_json(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
    return {}

def save_json(data, filepath):
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")

# Rebuild FAISS from JSON at startup to exclude previously scraped content.

# Load persistent data
chat_history = load_json(CHAT_HISTORY_FILE)
context_data = load_json(CONTEXT_JSON_FILE)

# Initialize as dict if not already
if not isinstance(chat_history, dict):
    chat_history = {}
if not isinstance(context_data, dict):
    context_data = {}

# --- Load JSON context into FAISS on startup ---
def initialize_json_context():
    """Load pre-populated JSON context into FAISS on startup"""
    global vector_store
    if not context_data:
        logger.warning("⚠️ No pre-populated context data found")
        return
    
    logger.info("📥 Loading pre-populated JSON context into FAISS...")
    
    documents = []
    for key, content in context_data.items():
        if isinstance(content, str) and content.strip():
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            texts = splitter.split_text(content)
            for chunk in texts:
                documents.append(Document(
                    page_content=chunk, 
                    metadata={"source": "json", "key": key}
                ))
    
    if documents:
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        logger.info(f"Loaded {len(documents)} JSON context chunks into FAISS")

# Initialize JSON context on startup
initialize_json_context()

# --- Helper Functions ---
def retrieve_and_answer(query, session_id):
    context = ""
    sources = []

    # Search the JSON context indexed in FAISS first
    if vector_store is not None and getattr(vector_store.index, "ntotal", 0) > 0:
        try:
            results = vector_store.similarity_search(query, k=3)
            for doc in results:
                source = f"Knowledge Base: {doc.metadata.get('key', 'General Info')}"

                context += f"From {source}:\n{doc.page_content}\n\n"
                sources.append(source)
        except Exception as e:
            logger.error(f"Error in FAISS search: {e}")

    # If no context found, try direct JSON lookup as fallback
    if not context.strip():
        logger.info("No FAISS results, checking JSON context directly...")
        query_words = set(query.lower().split())
        for key, content in context_data.items():
            if isinstance(content, str) and any(word in content.lower() for word in query_words if len(word) > 3):
                context += f"From Knowledge Base: {key}\n{content}\n\n"
                sources.append(f"Knowledge Base: {key}")
                break

    if not context.strip():
        return "🤖 No relevant data found in our knowledge base.", sources

    # Get conversation history for this session
    session_history = chat_history.get(session_id, [])
    conversation = ""
    history_limit = 5
    for q, a in session_history[-history_limit:]:
        conversation += f"User: {q}\nAI: {a}\n"
    conversation += f"User: {query}\nAI:"

    # Prepare prompt
    prompt = f"""
You are an AI assistant that answers questions based on available content.

## Instructions:
- Use ONLY the provided context to answer the user's question. 
- If the context is not enough, say you don't know instead of guessing. 
- Keep answers short, clear, and helpful (3–6 sentences max).  
- Maintain a helpful and professional tone.

## Context:
{context}

## Conversation History:
{conversation}
"""

    try:
        answer_obj = llm.invoke(prompt)

        # Convert AIMessage (or other object) to string
        if hasattr(answer_obj, "content"):
            answer = answer_obj.content
        else:
            answer = str(answer_obj)

        # Update chat history
        if session_id not in chat_history:
            chat_history[session_id] = []
        chat_history[session_id].append([query, answer])
        save_json(chat_history, CHAT_HISTORY_FILE)

        return answer, sources

    except Exception as e:
        error_msg = f"❌ Failed to generate answer: {str(e)}. Ensure the Ollama server is running and the Mistral model is available."
        logger.error(error_msg)
        return error_msg, []


# --- API Endpoints ---
@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question based on stored content"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    answer, sources = retrieve_and_answer(request.question, session_id)
    
    return ChatResponse(
        answer=answer,
        session_id=session_id,
    )

@app.post("/reset", response_model=ResetResponse)
async def reset_all():
    """Reset FAISS index and chat history (but keep JSON context)"""
    global vector_store
    
    vector_store = None
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
    
    chat_history.clear()
    
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)
    
    # Re-initialize JSON context after reset
    initialize_json_context()
    
    return ResetResponse(message="✅ FAISS index and chat history cleared! JSON context reloaded.")

@app.get("/context-data")
async def get_context_data():
    """Get all pre-populated JSON context data"""
    return {"context_data": context_data}

@app.get("/chat-history/{session_id}", response_model=SessionHistoryResponse)
async def get_chat_history(session_id: str):
    """Get chat history for a specific session"""
    history = chat_history.get(session_id, [])
    return SessionHistoryResponse(session_id=session_id, history=[{"question": q, "answer": a} for q, a in history])

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ollama_status = "unknown"
    try:
        test_response = llm.invoke("Say 'hello'")
        ollama_status = "healthy" if test_response else "unhealthy"
    except Exception as e:
        ollama_status = f"error: {str(e)}"
    
    faiss_doc_count = vector_store.index.ntotal if vector_store and hasattr(vector_store, 'index') else 0
    
    return {
        "status": "healthy",
        "ollama": ollama_status,
        "faiss_loaded": vector_store is not None,
        "faiss_documents": faiss_doc_count,
        "context_keys_count": len(context_data)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
