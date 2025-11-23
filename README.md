# JAX + LLM Routing System  
### **An End-to-End AI Customer Support Automation Pipeline**

This project implements a **production-grade AI support system** that mirrors real-world workflows used by ML Platform, Applied ML, and GenAI teams at companies like **Google, Meta, Apple, and NVIDIA**.

It combines:

- **JAX-based ML classifier**
- **Retrieval-Augmented Generation (RAG)**
- **Few-shot LLM support assistant**
- **LLM-as-Judge evaluation**
- **Safety guardrails**
- **Analytics & Observability Dashboard (Streamlit)**
- **Structured production logging**

---

## Overview

This system takes a user’s support message and performs:

1. **Routing** → Predict category using a trained JAX classifier  
2. **RAG Retrieval** → Find relevant KB documents  
3. **LLM Generation** → Produce a helpful grounded answer  
4. **Evaluation** → Score answer quality using an LLM judge  
5. **Guardrails** → Take safety actions if prediction is weak  
6. **Logging** → Store full interaction for analytics  
7. **Dashboard** → Visualize metrics, charts, and insights  

This design resembles real customer-support AI pipelines deployed in enterprise production.

---

## Features

### **JAX-Based Ticket Router**
- Multi-layer neural network built with **Flax + Optax**
- Predicts categories: *Billing, Technical, Account, Refund, Bug*
- Outputs confidence scores
- Trained end-to-end using OpenAI embeddings

---

### **RAG: Retrieval-Augmented Generation**
- Uses semantic embeddings to match top support articles  
- Filters by category for precision  
- Returns grounded context for the LLM  

---

### **LLM Support Assistant**
- Uses **GPT-4o-mini** for inference  
- Category-aware few-shot prompting  
- Optionally includes RAG snippets in the final answer  
- Produces short, safe, professional responses  

---

### **LLM-as-Judge Evaluation**
Every answer is rated by a second model:

- Helpfulness  
- Accuracy  
- Clarity  
- Relevance  

Produces:

- **score (0–1)**
- **reasoning/explanation**

This mimics human QA evaluation pipelines.

---

### **Guardrails**
Automatic safety actions:

| Condition | Action |
|----------|--------|
| Router confidence < 0.55 | `ROUTER_LOW_CONFIDENCE_ESCALATE` |
| Judge score < 0.50 | `LLM_POOR_ANSWER_RETRY` |
| Else | `OK` |

---

### **Analytics Dashboard**
Built with **Streamlit**, includes:

- Raw interaction viewer  
- Category-wise judge scores  
- Router confidence trends  
- Guardrail action distribution  
- Download logs as CSV  

risky → escalate / retry

