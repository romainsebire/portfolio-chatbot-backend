# Portfolio Chatbot Backend

Agentic API backend for the AI portfolio chatbot at [romainsebire.fr](https://romainsebire.fr).  
Architecture: **MCP + LangGraph**: FastAPI -> ReAct Agent -> FastMCP RAG Server (Pinecone) -> Groq LLM Cascade.

---

## Video Demo

<!-- https://github.com/user-attachments/assets/ede303cc-b0dd-43df-91cc-097b38015ae0 -->

https://github.com/user-attachments/assets/ca02c0df-eca4-4e33-b990-d49da78bfd32

---

## Architecture

```
app/
├── main.py          # FastAPI entrypoint (CORS, routes, middlewares, exception handlers)
├── config.py        # pydantic-settings: typed configuration loaded from environment
├── security.py      # Rate limiting, security headers, input/output sanitization
├── models.py        # Pydantic request & response validation models
├── prompts.py       # Agent 2-pass system prompts (Pass 1 Router, Pass 2 Generator)
├── agent.py         # LangGraph ReAct agent (graph execution, memory, tool-calling)
├── mcp_server.py    # FastMCP server (RAG tools exposed over stdio)
├── rag.py           # RAG orchestrator (connects MCP -> Agent -> Response)
└── embeddings.py    # Pinecone Cloud Inference API client singleton
```

### Execution Flow

```
Frontend (romainsebire.fr)
    │ POST /chat {message, lang, thread_id}
    ▼
FastAPI (main.py): validation, sanitization, rate limiting
    │
    ▼
Orchestrator (rag.py): loads MCP tools, builds LangGraph agent
    │
    ▼
LangGraph Agent (agent.py): ReAct graph execution with memory
    │ <── System Prompt (prompts.py)
    │
    ├── [call_model] -> LLM decides to call a tool or reply directly
    │       │
    │       ▼
    ├── [tools] -> FastMCP Server (mcp_server.py)
    │       │       ├── retrieve_documents() -> Pinecone Cloud
    │       │       └── get_context_info()   -> Pinecone stats
    │       │
    │       ▼
    └── [call_model] -> LLM generates final response (Pass 2)
            │
            ▼
Output sanitization -> {reply, thread_id}
```

## Local Installation

```bash
# 1. Clone repository and enter directory
git clone <repo-url> && cd portfolio-chatbot-backend

# 2. Create Python virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Populate .env with API keys (Pinecone, Groq, etc.)

# 5. Launch local development server
uvicorn app.main:app --reload
```

## Vector Database Indexing

```bash
# Populate data/contexte.txt with personal portfolio data (CV, projects, experience)
# Run the embedding ingestion script (from local machine):
python scripts/embed.py
```

## Testing & Security Audits

```bash
# Run unit and security test suite (38 tests)
pytest tests/ -v

# Run dependency and static security audits
pip-audit
bandit -r app/
```

## API Specification

| Method | Endpoint  | Description                             |
|--------|-----------|-----------------------------------------|
| `POST` | `/chat`   | Send user message to chatbot agent      |
| `GET`  | `/health` | Health check endpoint                   |

**`POST /chat`**
```json
// Request
{ "message": "What are Romain's technical skills?", "lang": "en", "thread_id": "session-123" }

// Response
{ "reply": "Romain specializes in Python, Machine Learning, FastMCP, and FastAPI...", "thread_id": "session-123" }
```

## Security Features

| Feature                     | Implementation                                          |
|-----------------------------|---------------------------------------------------------|
| Rate limiting               | `slowapi`: 10 requests/min per IP on `/chat`            |
| Daily token quota           | `DailyTokenLimiter`: 50,000 tokens/day per IP           |
| Strict CORS                 | Explicit allowed origins whitelist (no wildcard `*`)    |
| Input validation            | Pydantic: 250 max characters, payload limit <= 4KB     |
| Prompt injection defense    | Regex pattern matching + system prompt guardrails       |
| Output sanitization         | HTML/script tag stripping (anti-XSS)                    |
| Security headers            | HSTS, CSP, X-Frame-Options, X-Content-Type-Options, etc.|
| Secrets management          | `.env` gitignored, pydantic-settings                    |
| Traceability                | SHA-256 message hashing (no raw PII in logs)            |
| Log rotation                | `RotatingFileHandler`: 20 MB per file, max 2 backups    |
| Dependency auditing         | `pip-audit` in CI                                       |
| Static analysis             | `bandit` static security analysis in CI                 |
| Error handling              | Generic error details (no stack trace exposure)         |

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/deploy.yml`):

1. **`security`**: `pip-audit` + `bandit -r app/`
2. **`test`**: `pytest tests/ -v`
3. **`deploy`**: Trigger Render Deploy Hook (push on `main` only)

**Required GitHub Secret**: `RENDER_DEPLOY_HOOK_URL`

## Core Technologies

| Component | Technology | Description |
|-----------|------------|-------------|
| API | FastAPI | Async web framework, CORS, validation |
| Agent | LangGraph (ReAct) | Autonomous orchestration, stateful memory |
| Protocol | MCP (FastMCP) | Standardized interface for RAG tools |
| Vector DB | Pinecone | Cloud vector database |
| Embeddings | Pinecone Inference | Cloud embedding generation |
| LLM | Groq Cascade | Dynamic fallback cascade across Groq LLM models |
| Security | slowapi + custom middleware | Rate limiting, security headers, sanitization |
| CI/CD | GitHub Actions + Render | Automated test, audit, and deployment pipeline |

## Render Deployment

1. Create a **Web Service** on [render.com](https://render.com)
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Configure **Environment Variables** matching `.env.example`
5. Copy **Deploy Hook URL** and add as GitHub Secret `RENDER_DEPLOY_HOOK_URL`
