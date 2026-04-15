# Minnesota Legislation RAG

This project explores retrieval-augmented generation (RAG) and summarization over jargon-heavy legislative documents. The goal is to determine whether a RAG system performs better when it retrieves not only from original bill text, but also from plain-English summaries generated from those bills. The project corpus is based on Minnesota housing legislation.

Two systems will be compared:

1. A baseline RAG pipeline built only on original bill text
2. A summary-augmented RAG pipeline built on original bill text plus generated summaries

## Project Goal

This project aims to answer the following question:

**Does adding plain-English summaries to a legislative RAG pipeline improve the clarity, relevance, and usefulness of responses compared to retrieval over original bill text alone?**

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd mn-legislation-rag
```

Planned setup:

1. Create and activate a virtual environment
2. Install the project in editable mode
3. Copy `.env.example` to `.env`
4. Add your API key and local paths
5. Run ingestion and preprocessing scripts

Example install:

```bash
pip install -e ".[dev]"
```

## Running Main Scripts

Work in progress.

Planned script flow:

Extract bill text from PDFs
Clean and chunk the documents
Generate plain-English summaries
Build retrieval indexes
Run baseline RAG
Run summary-augmented RAG
Evaluate and compare both systems

## Repo Structure
```
mn-legislation-rag/
├── README.md
├── .gitignore
├── pyproject.toml
├── .env.example
├── config/
│   ├── settings.yaml
│   └── prompts.yaml
├── data/
│   ├── raw/
│   │   ├── pdfs/
│   │   └── metadata/
│   ├── processed/
│   │   ├── text/
│   │   ├── cleaned/
│   │   ├── chunks/
│   │   └── summaries/
│   └── evaluation/
│       ├── questions.json
│       └── results/
├── notebooks/
│   └── exploration_only.ipynb
├── scripts/
│   ├── ingest_documents.py
│   ├── extract_text.py
│   ├── chunk_documents.py
│   ├── generate_summaries.py
│   ├── build_indexes.py
│   ├── run_baseline_rag.py
│   ├── run_summary_rag.py
│   └── evaluate_systems.py
├── src/
│   └── mn_legislation_rag/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── models/
│       ├── ingestion/
│       ├── summarization/
│       ├── retrieval/
│       ├── rag/
│       ├── evaluation/
│       └── utils/
├── tests/
└── docs/
```

## Current Status

## 
