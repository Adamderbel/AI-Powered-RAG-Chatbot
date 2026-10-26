# AI-Powered RAG Chatbot with JSON Knowledge Base

A **Retrieval-Augmented Generation (RAG)** chatbot built with **FastAPI**, **Ollama**, and **FAISS**, with a simple HTML, CSS, and JavaScript chat frontend.

The chatbot now uses local `context_data.json` as its knowledge source. Web scraping and the scraping endpoints have been removed.

## Features

- **Chat frontend**: Responsive chat page with suggested questions, loading feedback, and error messages.
- **JSON knowledge base**: Load project information from `context_data.json` beside `app.py`.
- **Semantic retrieval**: Hugging Face embeddings and FAISS search, with direct JSON lookup as a fallback.
- **Conversational AI**: Ollama (`phi3:mini`) answers using retrieved context and conversation history.
- **Session management**: Follow-up questions share a session; New Chat starts a fresh conversation without deleting existing history.
- **Persistent storage**: Chat history is saved to JSON. The FAISS index is rebuilt from the knowledge base at startup.

## How It Works

```text
Browser chat frontend
        |
POST /ask (question + session ID)
        |
FAISS search over JSON context -> JSON lookup fallback
        |
Retrieved context + conversation history
        |
Ollama (phi3:mini)
        |
Answer displayed in the chat
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | HTML, CSS, JavaScript (no build step) |
| Backend | FastAPI + Uvicorn |
| Language model | Ollama with phi3:mini |
| Embeddings | Hugging Face Sentence Transformers |
| Vector search | FAISS |
| Text processing | LangChain |
| Knowledge and history | JSON files |

## Project Structure

```text
AI-Powered-RAG-Chatbot/
|-- app.py                 # FastAPI backend and frontend serving
|-- context_data.json      # Local knowledge base
|-- static/
|   |-- index.html         # Chat page
|   |-- style.css          # Responsive styles
|   `-- chat.js            # Chat interactions and API requests
|-- faiss_index/           # Generated vector index
|-- chat_history.json      # Generated session history
`-- requirements.txt       # Python dependencies
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Adamderbel/AI-Powered-RAG-Chatbot.git
cd AI-Powered-RAG-Chatbot
```

### 2. Install Dependencies

```bash
python -m venv venv
```

Activate the environment in Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

### 3. Set Up Ollama

Install [Ollama](https://ollama.com/download), then pull the configured model:

```bash
ollama pull phi3:mini
```

If Ollama is not already running, start it in a separate terminal:

```bash
ollama serve
```

### 4. Prepare Your JSON Context

Keep `context_data.json` in the project root, beside `app.py`. The current loader indexes top-level entries with nonempty **string values**; nested objects and lists are not indexed.

```json
{
  "project_info": "Our project helps teams analyze market trends.",
  "core_features": "Interactive charts, predictive analytics, and automated reports."
}
```

Restart the app after editing this file to reload the context and rebuild the index.

### 5. Run the Demo

```bash
python app.py
```

Open **http://localhost:8000** for the chat frontend, or **http://localhost:8000/docs** for interactive API documentation. FastAPI serves the frontend directly; no separate frontend server or dependencies are needed.

Type a question or choose a suggestion. Press **Enter** to send, or **Shift + Enter** for a new line. Use **New Chat** to start a fresh session.

The first startup may download the embedding model (`sentence-transformers/all-MiniLM-L6-v2`), so an internet connection is needed for that download.

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Serve the chat frontend |
| POST | `/ask` | Ask a question using the JSON context |
| POST | `/reset` | Clear chat history and rebuild FAISS from the loaded JSON context |
| GET | `/context-data` | Return the loaded JSON knowledge base |
| GET | `/chat-history/{session_id}` | Return a session's conversation history |
| GET | `/health` | Check Ollama and report index/context status |

Example question request:

```http
POST /ask
Content-Type: application/json

{
  "question": "What is this project about?"
}
```

The response contains `answer` and `session_id`. Include the returned `session_id` in subsequent requests to continue that conversation.

**New Chat** only starts a new session. The `/reset` endpoint clears all stored conversation history and rebuilds the index; it keeps the JSON knowledge base.

## Troubleshooting

- **Cannot reach the frontend**: Start `python app.py` and open http://localhost:8000 rather than opening the HTML file directly.
- **Ollama connection failed**: Ensure Ollama is running and `ollama pull phi3:mini` has completed.
- **Missing knowledge base answers**: Check that `context_data.json` is valid JSON beside `app.py`, use top-level string values, and restart after edits.
- **Slow first response**: Model loading and initial embedding downloads can take time.
- **Memory issues**: Reduce the amount of context or adjust the model and chunk size in `app.py`.
