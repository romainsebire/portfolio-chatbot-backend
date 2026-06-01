# Portfolio Chatbot Backend

Backend API agentic pour le chatbot IA du portfolio [romainsebire.fr](https://romainsebire.fr).  
Architecture **MCP + LangGraph** : FastAPI → Agent ReAct → MCP Server RAG (Pinecone) → Groq LLM.

---

## Architecture

```
app/
├── main.py          # FastAPI (CORS, routes, middlewares, error handlers)
├── config.py        # pydantic-settings — chargement typé depuis .env
├── security.py      # Rate limiting, security headers, input/output sanitization
├── models.py        # Pydantic models (validation stricte)
├── prompts.py       # System prompts de l'agent (FR/EN)
├── agent.py         # Agent LangGraph ReAct (graphe, mémoire, tool-calling)
├── mcp_server.py    # Serveur MCP FastMCP (outils RAG exposés via MCP)
├── rag.py           # Orchestrateur (connecte MCP → Agent → Réponse)
└── embeddings.py    # Singleton sentence-transformers
evaluation/
├── dataset.py       # Dataset d'évaluation (questions/réponses attendues)
└── evaluate.py      # LLM-as-a-Judge (scoring automatique)
```

### Flux de traitement

```
Frontend (romainsebire.fr)
    │ POST /chat {message, lang, thread_id}
    ▼
FastAPI (main.py) — validation, sanitization, rate limiting
    │
    ▼
Orchestrateur (rag.py) — charge les outils MCP, construit l'agent
    │
    ▼
Agent LangGraph (agent.py) — graphe ReAct avec mémoire
    │ ◄── System Prompt (prompts.py)
    │
    ├── [call_model] → LLM décide d'appeler un outil
    │       │
    │       ▼
    ├── [tools] → MCP Server (mcp_server.py)
    │       │       ├── retrieve_documents() → Pinecone
    │       │       └── get_context_info()   → Pinecone stats
    │       │
    │       ▼
    └── [call_model] → LLM génère la réponse finale
            │
            ▼
Sanitization sortie → {reply, thread_id}
```

## Installation locale

```bash
# 1. Cloner et entrer dans le projet
git clone <repo-url> && cd portfolio-chatbot-backend

# 2. Créer un environnement virtuel
python -m venv .venv && source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# → Remplir .env avec tes clés Pinecone, Groq, etc.

# 5. Lancer le serveur de développement
uvicorn app.main:app --reload
```

## Alimentation de Pinecone

```bash
# Remplir data/contexte.txt avec tes données (CV, projets, PFE…)
# Puis lancer le script d'embedding (une seule fois, ou après mise à jour) :
python scripts/embed.py
```

**Prérequis** : un index Pinecone créé avec dimension `384` (pour `paraphrase-multilingual-MiniLM-L12-v2`).

## Tests

```bash
# Tests fonctionnels + sécurité
pytest tests/ -v

# Audit de sécurité
pip-audit
bandit -r app/
```

## API

| Méthode | Endpoint  | Description                    |
|---------|-----------|--------------------------------|
| `POST`  | `/chat`   | Envoyer un message au chatbot  |
| `GET`   | `/health` | Health check                   |

**`POST /chat`**
```json
// Request
{ "message": "Quelles sont les compétences de Romain ?", "lang": "fr", "thread_id": "session-123" }

// Response
{ "reply": "Romain maîtrise Python, FastAPI, JavaScript...", "thread_id": "session-123" }
```

## Évaluation

```bash
# Lancer l'évaluation automatique (LLM-as-a-Judge)
python -m evaluation.evaluate
```

Évalue le chatbot sur 15 questions avec scoring sur 3 axes :
- **Correctness** (40%) — exactitude factuelle
- **Completeness** (35%) — couverture de la question
- **Safety** (25%) — pas de fabrication, respect du cadre

## Sécurité

| Protection                  | Implémentation                                          |
|-----------------------------|---------------------------------------------------------|
| Rate limiting               | `slowapi` — 10 req/min par IP sur `/chat`               |
| CORS strict                 | Whitelist d'origines (pas de wildcard `*`)               |
| Validation d'entrée         | Pydantic : max 500 chars, lang restreint, body ≤ 1KB    |
| Prompt injection defense    | Filtrage regex + system prompt renforcé avec guardrails  |
| Output sanitization         | Strip HTML/script des réponses LLM (anti-XSS)           |
| Security headers            | HSTS, CSP, X-Frame-Options, X-Content-Type-Options, etc.|
| Secrets management          | `.env` gitignored, pydantic-settings, fail-fast          |
| Logging sûr                 | Hash SHA-256 des messages (pas de PII en clair)          |
| Audit dépendances           | `pip-audit` + Dependabot (scan hebdomadaire)             |
| Analyse statique            | `bandit` en CI                                           |
| Error handling              | Messages génériques (jamais de stack trace côté client)   |

## CI/CD

Pipeline GitHub Actions (`.github/workflows/deploy.yml`) :

1. **`security`** — `pip-audit` + `bandit -r app/`
2. **`test`** — `pytest tests/ -v`
3. **`deploy`** — Trigger Render Deploy Hook (uniquement sur push `main`)

**Secret GitHub requis** : `RENDER_DEPLOY_HOOK_URL`

## Technologies clés

| Composant | Technologie | Rôle |
|-----------|------------|------|
| API | FastAPI | Serveur web async, validation, CORS |
| Agent | LangGraph (ReAct) | Orchestration autonome, mémoire conversationnelle |
| Protocol | MCP (FastMCP) | Interface standardisée pour les outils RAG |
| Vector DB | Pinecone | Stockage vectoriel cloud (persistant, free tier) |
| Embeddings | sentence-transformers | Vectorisation des textes (multilingual) |
| LLM | Groq (Llama 3.3 70B) | Génération de réponses |
| Sécurité | slowapi + custom middlewares | Rate limiting, headers, sanitization |
| CI/CD | GitHub Actions + Render | Tests, audit sécurité, déploiement |

## Déploiement Render

1. Créer un **Web Service** sur [render.com](https://render.com)
2. **Build Command** : `pip install -r requirements.txt`
3. **Start Command** : `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Configurer les **Environment Variables** (mêmes que `.env.example`)
5. Copier le **Deploy Hook URL** → ajouter comme secret GitHub `RENDER_DEPLOY_HOOK_URL`
