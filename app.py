from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import os
import re
import shutil
import json
import uuid
from typing import List, Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configurations ---
FAISS_INDEX_PATH = "faiss_index"
SCRAPED_URLS_FILE = "scraped_urls.json"
CHAT_HISTORY_FILE = "chat_history.json"
CONTEXT_JSON_FILE = "./data/context_data.json" # Pre-populated context file

# --- Pydantic Models ---
class ScrapeRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    

class ScrapeResponse(BaseModel):
    message: str
    url: str

class ResetResponse(BaseModel):
    message: str

class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, str]]

# --- Initialize FastAPI ---
app = FastAPI(title="Web Scraper QA API", version="1.0.0")

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

# Load FAISS if exists
if os.path.exists(FAISS_INDEX_PATH):
    try:
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("✅ FAISS index loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load FAISS index: {e}")
        vector_store = None

# Load persistent data
scraped_urls = load_json(SCRAPED_URLS_FILE)
chat_history = load_json(CHAT_HISTORY_FILE)
context_data = load_json(CONTEXT_JSON_FILE)

# Initialize as dict if not already
if not isinstance(scraped_urls, list):
    scraped_urls = []
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
        if vector_store is None:
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(FAISS_INDEX_PATH)
            logger.info(f"✅ Loaded {len(documents)} JSON context chunks into FAISS")
        else:
            # Check if JSON context is already loaded to avoid duplicates
            existing_sources = set()
            try:
                # Sample search to check existing content
                sample_results = vector_store.similarity_search("test", k=10)
                for doc in sample_results:
                    if doc.metadata.get("source") == "json":
                        existing_sources.add(doc.metadata.get("key"))
            except:
                pass
            
            # Only add new JSON content
            new_documents = [doc for doc in documents if doc.metadata["key"] not in existing_sources]
            if new_documents:
                vector_store.add_documents(new_documents)
                vector_store.save_local(FAISS_INDEX_PATH)
                logger.info(f"✅ Added {len(new_documents)} new JSON context chunks to FAISS")

# Initialize JSON context on startup
initialize_json_context()

# --- Helper Functions ---
def validate_url(url):
    if not url:
        return None
    if not re.match(r'^https?://', url):
        url = f"https://{url}"
    return url

def scrape_website(url):
    try:
        logger.info(f"🌍 Scraping website: {url}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            return f"⚠️ Failed to fetch {url}: HTTP {response.status_code}"

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text from multiple tag types
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th"])
        if not paragraphs:
            return f"⚠️ No content found on {url}"

        text = " ".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        if not text.strip():
            return f"⚠️ Empty content extracted from {url}"

        # Clean up text
        text = re.sub(r'\s+', ' ', text)
        return text[:10000]
    
    except requests.exceptions.InvalidSchema:
        return f"❌ Invalid URL: {url}"
    except requests.exceptions.Timeout:
        return f"❌ Request timed out for {url}"
    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

def store_in_faiss(text, url):
    global vector_store
    logger.info("📥 Storing scraped data in FAISS...")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"url": url, "source": "web"}) for chunk in texts]

    if vector_store is None:
        vector_store = FAISS.from_documents(documents, embeddings)
    else:
        vector_store.add_documents(documents)

    vector_store.save_local(FAISS_INDEX_PATH)
    return "✅ Data stored successfully!"

def retrieve_and_answer(query, session_id):
    context = ""
    sources = []

    # Always try FAISS first (contains both scraped and JSON data)
    if vector_store is not None and getattr(vector_store.index, "ntotal", 0) > 0:
        try:
            results = vector_store.similarity_search(query, k=3)
            for doc in results:
                source_type = doc.metadata.get('source', 'unknown')
                if source_type == 'web':
                    source = doc.metadata.get('url', 'Scraped Website')
                else:
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
        return "🤖 No relevant data found in our knowledge base or scraped content.", sources

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
    return {"message": "Web Scraper QA API", "status": "running"}

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_website_endpoint(request: ScrapeRequest):
    """Scrape a website and store its content"""
    url = validate_url(request.url)
    if not url:
        raise HTTPException(status_code=400, detail="Invalid URL")
    
    content = scrape_website(url)
    if "⚠️" in content or "❌" in content:
        raise HTTPException(status_code=400, detail=content)
    
    store_message = store_in_faiss(content, url)
    
    # Update scraped URLs
    if url not in scraped_urls:
        scraped_urls.append(url)
        save_json(scraped_urls, SCRAPED_URLS_FILE)
    
    return ScrapeResponse(message=store_message, url=url)

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
    """Reset FAISS index, scraped URLs, and chat history (but keep JSON context)"""
    global vector_store
    
    vector_store = None
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
    
    scraped_urls.clear()
    chat_history.clear()
    
    if os.path.exists(SCRAPED_URLS_FILE):
        os.remove(SCRAPED_URLS_FILE)
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)
    
    # Re-initialize JSON context after reset
    initialize_json_context()
    
    return ResetResponse(message="✅ FAISS index, scraped URLs, and chat history cleared! JSON context reloaded.")

@app.get("/scraped-urls")
async def get_scraped_urls():
    """Get list of all scraped URLs"""
    return {"scraped_urls": scraped_urls}

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
        "scraped_urls_count": len(scraped_urls),
        "context_keys_count": len(context_data)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)