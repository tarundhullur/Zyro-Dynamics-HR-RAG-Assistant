# 📋 Zyro Dynamics HR Help Desk

An AI-powered Retrieval-Augmented Generation (RAG) chatbot designed to assist employees of Zyro Dynamics Pvt. Ltd. with internal HR policy queries. The application leverages a secure, cached vector pipeline to pull relevant information exclusively from company PDFs and provides accurate, concise answers while filtering out unrelated Out-of-Scope (OOS) questions.

---

## 🚀 Features

*   **RAG Pipeline**: Loads and processes HR policy PDFs dynamically using `PyPDFDirectoryLoader` and chunks them intelligently.
*   **Vector Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face to generate local text embeddings stored in a lightweight `FAISS` vector database.
*   **Topic Guardrails**: Employs a zero-shot intent classifier powered by `Llama-3.1-8b-instant` to reject out-of-scope queries (e.g., general trivia, weather) with a standardized polite refusal.
*   **Observability**: Integrated natively with **LangSmith** for deep tracking, tracing, and debugging of LLM and retrieval performance.
*   **User-Friendly Interface**: Built entirely with a sleek, conversational Streamlit UI.

---

## 🛠️ Tech Stack & Dependencies

*   **Frontend UI**: Streamlit
*   **LLM & Orchestration**: LangChain, LangChain Community, ChatGroq (`llama-3.1-8b-instant`)
*   **Vector DB & Embeddings**: FAISS (CPU), Hugging Face Embeddings
*   **Monitoring/Tracing**: LangSmith

---

## 📦 Project Structure

```text
├── data/                  # Drop your HR policy PDF files here
├── app.py                 # Main Streamlit application source code
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
