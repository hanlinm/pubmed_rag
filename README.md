# 🧬 Biomedical Literature Assistant

A retrieval-augmented generation (RAG) pipeline that answers research questions 
grounded in PubMed scientific literature. Built with LangChain, ChromaDB, and 
the OpenAI API, deployed as an interactive Streamlit application.

🔗 **[Live Demo](https://pubmedrag-biomedical-literature-assistant.streamlit.app/)**

---

## Overview

Traditional LLM chatbots hallucinate when asked about specific scientific findings. 
This application solves that by retrieving relevant abstracts from an indexed corpus 
of PubMed papers and grounding every answer in real literature — with citations.

Ask a question → retrieve the most relevant research chunks → generate a cited, 
grounded answer. No hallucination, no guessing.

---

## Features

- 🔍 **Semantic search** over 100+ PubMed abstracts using vector similarity
- 📄 **Source citations** — every answer includes paper titles and PubMed links
- 💬 **Conversational interface** with persistent chat history
- ⚡ **Streaming responses** token by token
- 🧠 **Grounded generation** — LLM is explicitly instructed not to answer outside retrieved context

---

## Covered Research Areas

- Protein structure prediction & deep learning
- Generative AI for drug discovery
- GLP-1 receptor agonists & metabolic disease
- Biomedical NLP & transformer models
- Cancer immunotherapy & machine learning
- CRISPR & gene editing
- Single cell RNA sequencing
- Clinical trial outcome prediction
- Antibiotic resistance prediction
- Protein language models & therapeutics

---

## Architecture
```
User Question
      │
      ▼
OpenAI Embeddings          ← converts question to vector
      │
      ▼
ChromaDB Vector Store      ← finds top-k most similar chunks
      │
      ▼
Retrieved Abstracts        ← real PubMed papers with metadata
      │
      ▼
GPT-4o-mini + Prompt       ← generates grounded, cited answer
      │
      ▼
Streamlit UI               ← streams response to user
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangChain (LCEL) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | ChromaDB |
| LLM | GPT-4o-mini |
| Data source | PubMed E-utilities API |
| UI | Streamlit |

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/hanlinm/pubmed-rag.git
cd pubmed-rag
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv rag_env
source rag_env/bin/activate  # Windows: rag_env\Scripts\activate
pip install -r requirements.txt
```

**3. Set up environment variables**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key
NCBI_EMAIL=your@email.com
```

**4. Fetch papers and build the vector store**
```bash
python ingest.py
```

**5. Launch the app**
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## Project Structure
```
pubmed_rag/
├── ingest.py        # fetches PubMed papers and builds ChromaDB vector store
├── query.py         # retrieval chain and LLM generation logic
├── app.py           # Streamlit chat interface
├── requirements.txt
└── .env             # not committed — see above
```

---

## Key Design Decisions

**Chunk size of 800 with 100 token overlap** — scientific abstracts are dense. 
Smaller chunks lose context, larger chunks reduce retrieval precision. 800 tokens 
balances both for abstract-length text.

**temperature=0 on the LLM** — scientific question answering requires 
deterministic, factual responses. Temperature 0 eliminates creative variation 
that could introduce inaccuracies.

**text-embedding-3-small over text-embedding-3-large** — benchmarking showed 
negligible quality difference on abstract-length biomedical text at 5x lower cost.

**Explicit grounding in the prompt** — the system prompt instructs the LLM to 
answer only from retrieved context and explicitly say so if the answer isn't 
there. This is the primary hallucination guard.

---

## Future Improvements

- [ ] Hybrid search (BM25 + vector) for better retrieval of exact gene/drug names
- [ ] Reranking with Cohere to improve chunk relevance ordering
- [ ] Semantic chunking for higher faithfulness scores
- [ ] Expanded corpus — full text papers via PubMed Central
- [ ] LangSmith tracing for retrieval quality monitoring
- [ ] User-configurable topic selection

---

## About

Built as part of a portfolio project to demonstrate applied LLM engineering 
skills including RAG architecture, vector database design, prompt engineering, 
and production deployment. Domain focus reflects 5+ years of professional 
experience in computational biology and drug discovery.
