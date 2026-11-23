# Portfolio Chatbot Backend

> **Agentic RAG API backend powering the conversational assistant on [romainsebire.fr](https://romainsebire.fr), built with FastAPI, LangGraph, FastMCP, Pinecone, and a resilient Groq LLM cascade**

[![CI/CD](https://github.com/romainsebire/portfolio-chatbot-backend/actions/workflows/deploy.yml/badge.svg)](https://github.com/romainsebire/portfolio-chatbot-backend/actions/workflows/deploy.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-romainsebire.fr-blue)](https://romainsebire.fr)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![FastMCP](https://img.shields.io/badge/FastMCP-MCP%20Server-6B46C1)](https://gofastmcp.com/)

---

## Overview

This service powers the bilingual (French and English) AI assistant integrated into [romainsebire.fr](https://romainsebire.fr). It answers visitor and recruiter questions about projects, skills, and experience by retrieving structured context from a Pinecone vector database through a FastMCP tool server.

The backend is designed for production reliability, combining automatic failover across multiple Groq models, per-IP quota and rate limits, prompt injection filtering, and automated CI/CD security checks. A high-level case study is also available on the [Portfolio RAG Chatbot project page](https://romainsebire.fr/en/projects/portfolio-rag-chatbot).

---

## Video Demo

https://github.com/user-attachments/assets/ca02c0df-eca4-4e33-b990-d49da78bfd32

---

## Architecture

```text
app/
├── main.py          # FastAPI entrypoint (CORS, routes, middlewares, exception handlers)
├── config.py        # pydantic-settings: typed configuration loaded from environment
├── security.py      # Rate limiting, security headers, input/output sanitization
├── models.py        # Pydantic request & response validation models
├── prompts.py       # Agent 2-pass system prompts (Pass 1 Router, Pass 2 Generator)
├── agent.py         # LangGraph ReAct agent (graph execution, memory, tool-calling)
├── mcp_server.py    # FastMCP server (RAG tools exposed over stdio)
├── rag.py           # RAG orchestrator (connects MCP, Agent, and Response)
└── embeddings.py    # Pinecone Cloud Inference API client singleton
evaluation/
├── dataset.py       # Golden benchmark dataset (23 test cases across 4 difficulty tiers)
└── evaluate.py      # Retrieval & end-to-end RAG evaluation CLI (with optional LLM-as-a-Judge)
scripts/
└── embed.py         # Semantic chunking & Pinecone vector ingestion script
```

### Execution Flow

```text
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
    │ <-- System Prompt (prompts.py)
    │
    ├── [call_model] Pass 1: LLM decides to call a tool or reply directly
    │       │
    │       ▼
    ├── [tools] FastMCP Server (mcp_server.py)
    │       │       ├── retrieve_documents() : Pinecone Cloud
    │       │       └── get_context_info()   : Pinecone stats
    │       │
    │       ▼
    └── [call_model] Pass 2: LLM generates final response (tools unbound)
            │
            ▼
Output sanitization : {reply, thread_id}
```

---

## Evaluation & Benchmark

A dedicated benchmark suite (`evaluation/`) evaluates both retrieval quality and end-to-end agent answers without relying on guesswork. Prefixing each semantic chunk with its section title raises vector retrieval recall within the top 3 results to **100%**.

End-to-end quality is evaluated across **23 questions** covering simple queries, technical project details, multi-turn follow-ups, and guardrails (off-topic queries, privacy, prompt injection), combining deterministic checks with a fixed LLM judge (`GPT-OSS 120B`):

| Evaluated Model | Deterministic Checks | Overall Score (Judge) | Safety |
|---|:---:|:---:|:---:|
| **Qwen 3.8 27B** | **91.3%** | **8.6 / 10** | **10 / 10** |
| GPT-OSS 120B | 87.0% | 8.1 / 10 | 10 / 10 |
| GPT-OSS 20B | 82.6% | 8.0 / 10 | 10 / 10 |

Based on these results, **Qwen 3.8 27B** serves as the primary model for the clarity of its summaries and adherence to guardrails, backed by **GPT-OSS 120B** and **GPT-OSS 20B** in the automatic fallback cascade.

---

## Local Installation

```bash
# 1. Clone repository and enter directory
git clone https://github.com/romainsebire/portfolio-chatbot-backend.git && cd portfolio-chatbot-backend

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

## Testing, Security Audits & RAG Benchmark

```bash
# Run unit and security test suite (44 tests)
pytest tests/ -v

# Run Pinecone retrieval benchmark (0 Groq LLM tokens consumed)
python -m evaluation.evaluate --retrieval-only

# Run full end-to-end Agentic RAG benchmark on a single model with rate-limit pacing (35s delay)
python -m evaluation.evaluate --model qwen/qwen3.8-27b --delay 35 --judge

# Run dependency and static security audits
pip-audit
bandit -r app/
```

---

## API Specification

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Send user message to chatbot agent |
| `GET` | `/health` | Health check endpoint |

**`POST /chat`**
```json
// Request
{ "message": "What are Romain's technical skills?", "lang": "en", "thread_id": "session-123" }

// Response
{ "reply": "Romain specializes in Python, Machine Learning, FastMCP, and FastAPI...", "thread_id": "session-123" }
```

---

## Security Features

| Feature | Implementation |
|---|---|
| Rate limiting | `slowapi`: 10 requests/min per IP on `/chat` |
| Daily token quota | `DailyTokenLimiter`: 50,000 tokens/day per IP |
| Strict CORS | Explicit allowed origins whitelist (no wildcard `*`) |
| Input validation | Pydantic: 250 max characters, payload limit <= 4KB |
| Prompt injection defense | Regex pattern matching + system prompt guardrails |
| Output sanitization | HTML/script tag stripping (anti-XSS) |
| Security headers | HSTS, CSP, X-Frame-Options, X-Content-Type-Options, etc. |
| Secrets management | `.env` gitignored, `pydantic-settings` |
| Traceability | SHA-256 message hashing (no raw PII in logs) |
| Log rotation | `RotatingFileHandler`: 20 MB per file, max 2 backups |
| Dependency auditing | `pip-audit` in CI |
| Static analysis | `bandit` static security analysis in CI |
| Error handling | Generic error details (no stack trace exposure) |

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/deploy.yml`):

1. **`security`**: `pip-audit` + `bandit -r app/`
2. **`test`**: `pytest tests/ -v`
3. **`deploy`**: Trigger Render Deploy Hook (push on `main` only)

**Required GitHub Secret**: `RENDER_DEPLOY_HOOK_URL`

---

## Core Technologies

| Component | Technology | Description |
|---|---|---|
| API | FastAPI | Async web framework, CORS, validation |
| Agent | LangGraph (ReAct) | Two-pass autonomous orchestration and stateful memory |
| Protocol | MCP (FastMCP) | Standardized interface for RAG tools |
| Vector DB | Pinecone | Cloud vector database |
| Embeddings | Pinecone Inference | Cloud embedding generation (`multilingual-e5-large`) |
| LLM | Groq Cascade | Configurable fallback cascade (`qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`) |
| Security | slowapi + custom middleware | Rate limiting, security headers, sanitization |
| CI/CD | GitHub Actions + Render | Automated test, audit, and deployment pipeline |

---

## Render Deployment

1. Create a **Web Service** on [render.com](https://render.com)
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Configure **Environment Variables** matching `.env.example`
5. Copy **Deploy Hook URL** and add as GitHub Secret `RENDER_DEPLOY_HOOK_URL`
