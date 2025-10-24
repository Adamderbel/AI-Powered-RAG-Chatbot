# 🧠 AI-Powered RAG Chatbot with Web Scraping & Knowledge Base

A sophisticated **Retrieval-Augmented Generation (RAG)** chatbot that combines web scraping capabilities with pre-populated knowledge base content. Built with **FastAPI**, **Ollama**, and **FAISS**, this system provides intelligent, context-aware responses by leveraging both dynamically scraped web content and static JSON knowledge sources.

> 📌 **Key Use Case**: Create a comprehensive AI assistant that understands both your website content and existing knowledge base, providing accurate, sourced answers while maintaining conversation history.

---

## 🚀 Features

- **🌐 Web Scraping**: Automatically extract and process content from any website
- **📚 Dual Knowledge Sources**: Combine scraped web data with pre-populated JSON context
- **🔍 Smart Retrieval**: FAISS vector search for relevant content matching
- **💬 Conversational AI**: Ollama-powered LLM with session-based chat history
- **🔄 Persistent Storage**: Maintain scraped URLs, FAISS index, and chat sessions
- **🎯 Context-Aware**: Uses both conversation history and retrieved content
- **⚡ FastAPI Backend**: High-performance REST API with CORS support
- **🔒 Session Management**: Isolated chat histories for different users/sessions

---

## 🏗️ System Architecture

```
User Query 
    ↓
FastAPI Endpoint (/ask)
    ↓
FAISS Vector Search → JSON Context Fallback
    ↓
Context + Conversation History
    ↓
Ollama LLM (phi3:mini)
    ↓
Response with Sources
```

---

## 📦 Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend Framework** | FastAPI |
| **LLM** | Ollama with Phi-3 model |
| **Embeddings** | HuggingFace Sentence Transformers |
| **Vector Database** | FAISS |
| **Web Scraping** | Requests + BeautifulSoup |
| **Text Processing** | LangChain |
| **Data Persistence** | JSON files |

---

## 🗂️ Project Structure

```
📁 rag-chatbot/
├── app.py                          # FastAPI application
├── data/
│   └── context_data.json            # Pre-populated knowledge base
├── faiss_index/                     # Vector store (auto-generated)
├── scraped_urls.json               # Tracked URLs (auto-generated)
├── chat_history.json               # Session histories (auto-generated)
└── requirements.txt                # Dependencies
```

---

## ⚙️ Core Components

### 1. **Dual Knowledge Sources**
- **Web Content**: Dynamically scraped from URLs
- **JSON Context**: Pre-loaded from `context_data.json`
- **Smart Merging**: FAISS combines both sources seamlessly

### 2. **Intelligent Retrieval**
- Semantic search using FAISS vector similarity
- Fallback to direct JSON lookup when needed
- Source tracking for response attribution

### 3. **Conversation Management**
- UUID-based session tracking
- Persistent chat history
- Context-aware prompting with history

### 4. **Content Processing**
- Automatic text chunking (500 chars with 100 overlap)
- HTML cleaning and normalization
- Duplicate prevention

---

## 🖥️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Adamderbel/AI-Powered-RAG-Chatbot.git
cd AI-Powered-RAG-Chatbot
```

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Up Ollama
1. Install [Ollama](https://ollama.com/download) (supports Linux, macOS, Windows).
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Pull and run the phi3:mini model (or any Ollama-supported model):
   ```bash
   ollama pull phi3:mini
   ollama run phi3:mini
   ```

### 4. Run the App
```bash
python app.py
```

---

## 📡 API Endpoints

### 🔍 **Scrape Website**
```http
POST /scrape
Content-Type: application/json

{
  "url": "https://example.com"
}
```

### 💬 **Ask Question**
```http
POST /ask
Content-Type: application/json

{
  "question": "What services do you offer?",
  "session_id": "optional-session-uuid"
}
```

### 🗑️ **Reset System**
```http
POST /reset
```

### 📊 **Get Scraped URLs**
```http
GET /scraped-urls
```

### 💾 **Get Chat History**
```http
GET /chat-history/{session_id}
```

### 🩺 **Health Check**
```http
GET /health
```
---

## 🐛 Troubleshooting

### Common Issues:

1. **Ollama Connection Failed**
   ```bash
   # Ensure Ollama is running
   ollama serve
   ```

2. **FAISS Loading Errors**
   ```bash
   # Reset the vector store
   curl -X POST http://localhost:8000/reset
   ```

3. **Memory Issues**
   - Reduce chunk size in `CharacterTextSplitter`
   - Limit number of scraped URLs
   - Use smaller embedding model

4. **Scraping Failures**
   - Check URL accessibility
   - Verify network connectivity
   - Review website robots.txt

---





