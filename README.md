JAX + LLM Routing System

A Production-Grade Customer Support Routing, RAG, LLM-Generation & Evaluation Framework

This project demonstrates a full end-to-end AI support automation pipeline, combining:

JAX-based ML classifier for category routing

RAG (Retrieval-Augmented Generation) for grounded answers

Few-shot LLM support assistant

LLM-as-judge automatic evaluation

Guardrails for safety & reliability

Analytics dashboard with charts & insights

Structured CSV logging for production observability

This system closely mirrors how modern AI/ML teams at FAANG-level companies design support automation systems.

Live Capabilities

✔ JAX multi-layer router (Billing, Technical, Account, Refund, Bug)
✔ RAG retrieval from semantic Knowledge Base
✔ Category-aware few-shot LLM generation
✔ LLM-as-judge evaluation (score + reasoning)
✔ Guardrails (low router confidence, bad LLM answer, escalation flags)
✔ Full interaction logging
✔ Chart-based Evaluation Dashboard (Streamlit)

Project Structure
jax-llm-routing/
│
├── pipeline/
│   ├── preprocess.py              # Cleaning + preparing ticket dataset
│   ├── compute_embeddings.py      # Embedding generation (train + KB)
│   ├── orchestrator.py            # FULL pipeline (router → RAG → LLM → judge → guardrails)
│   ├── judge.py                   # LLM-as-judge evaluation
│   ├── logger.py                  # CSV logging
│
├── models/
│   ├── jax_router.py              # JAX Flax multi-layer classifier
│   ├── train_router.py            # Training loop + saved params
│   └── saved_params/              # router.pkl (after training)
│
├── rag/
│   ├── index_docs.py              # Build KB embeddings + metadata
│   ├── retriever.py               # Semantic search (cosine similarity)
│
├── llm/
│   └── use_ft_model.py            # Few-shot prompt-based answer generator
│
├── dashboard/
│   └── app.py                     # Streamlit demo + analytics UI
│
├── data/
│   ├── raw_tickets.csv
│   ├── tickets_small.csv
│   └── knowledge_base.csv
│
├── logs/
│   └── interactions.csv           # Auto-generated
│
├── config.py                      # Global model/settings paths
└── README.md

System Architecture
───────────────────────────────────────────────────────────────
                     FULL PIPELINE (Orchestrator)
───────────────────────────────────────────────────────────────

         User Message
                 │
                 ▼
        ┌──────────────────┐
        │  JAX Router       │  → Predicts Category + Confidence
        └──────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │   RAG Retriever   │  → Top-k similar KB documents
        └──────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  LLM Generator    │  → Few-shot + RAG grounded answer
        │  (GPT-4o-mini)    │
        └──────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  LLM-as-Judge     │  → Score (0–1) + reasoning
        └──────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │   Guardrails      │  → LOW_CONF, BAD_ANSWER, OK
        └──────────────────┘
                 │
                 ▼
            Logs (CSV)
───────────────────────────────────────────────────────────────

Key Features (Detailed)
1. JAX Router (Category Classifier)

2-layer neural network using Flax + Optax

Trained on embeddings of support tickets

Predicts categories (Billing, Technical, Account, etc.)

Outputs:

predicted category

router confidence

Saved as models/saved_params/router.pkl

2. Few-Shot LLM Answering

Uses GPT-4o-mini with:

system prompt

few-shot category-specific examples

optional RAG context

Ensures:

consistent tone

helpful, safe responses

no hallucinations due to grounding

3. RAG Retrieval (Knowledge Base Search)

Embeds KB articles using OpenAI embeddings

Computes cosine similarity between:

query embedding

KB embeddings

Returns top-k articles filtered by category

Used to ground LLM answers with real data

4. LLM-as-Judge Evaluation

A second LLM evaluates the model’s final answer:

helpfulness

relevance

accuracy

clarity

Produces:

numeric score 0–1

reasoning explanation

5. Guardrails

Triggered automatically:

Condition	Action
router_confidence < 0.55	ROUTER_LOW_CONFIDENCE_ESCALATE
judge_score < 0.5	LLM_POOR_ANSWER_RETRY
else	OK

Outputs are logged for monitoring.

6. Structured Logging

Every interaction is appended to:

logs/interactions.csv


With fields:

timestamp

user_message

predicted_category

router_confidence

judge_score

judge_reasoning

guardrail_action

guardrail_notes

7. Streamlit Dashboard
Chat / Demo Page

Enter user message

Shows:

router prediction

RAG KB hits

LLM final output

judge scoring

guardrail action

Evaluation Dashboard

Raw logs

Total interactions

Mean judge score

Mean router confidence

Per-category judge scores

Guardrail action distribution

Download CSV button

Production insights charts

Industry-Level Insights (Built-In)

The dashboard provides insights like:

Category-wise success rate

Judge score distribution

Average guardrail triggers

Model drift indicators

User message analytics

RAG hit quality & coverage

These are intentionally aligned with real enterprise GenAI observability workflows.

Installation
1. Clone the repo
git clone https://github.com/yourusername/jax-llm-routing.git
cd jax-llm-routing

2. Create environment
python -m venv jaxwin
source jaxwin/bin/activate  # Windows: jaxwin\Scripts\activate

3. Install requirements
pip install -r requirements.txt

4. Set your API key

Create .env:

OPENAI_API_KEY=your_key_here

Running the System
1. Build Knowledge Base index
python -m rag.index_docs

2. Train the router
python -m models.train_router

3. Launch dashboard
streamlit run dashboard/app.py


Example Interaction

User:

I was charged double for my subscription.

Router: Billing (0.96 confidence)

RAG KB Hits:

Billing – Cancel subscription

Billing – Trial period info

LLM Answer:
Helpful grounded explanation with refund instructions.

Judge Score:
0.92 — “clear, correct, grounded”

Guardrail:
OK


Contact

If you use this project or want improvements, feel free to reach out.