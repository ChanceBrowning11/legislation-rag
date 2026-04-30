# Minnesota Legislation RAG

This project explores retrieval-augmented generation (RAG) and summarization over jargon-heavy legislative documents. The goal is to determine whether a RAG system performs better when it retrieves not only from original bill text, but also from plain-English summaries generated from those bills. The project corpus is based on Minnesota housing legislation.

Two systems will be compared:

1. **Baseline RAG**: retrieval over original bill text only
2. **Summary-Augmented RAG**: retrieval over original bill text plus plain-English bill summaries generated from the source text

The core research question is:

**Does adding plain-English summaries to a legislative RAG pipeline improve the clarity, relevance, and usefulness of answers compared to retrieval over original bill text alone?**

---

## Project Overview

Minnesota legislative bills are often dense, formal, and difficult for non-experts to read. This project builds an end-to-end pipeline that:

- extracts and cleans legislative bill text from PDFs
- chunks bills into retrieval-ready records
- generates plain-English summaries for each bill
- builds two vector indexes for comparison
- runs baseline and summary-augmented RAG pipelines
- evaluates both systems on a shared question set

The current corpus is focused on **Minnesota housing bills**, using a starter corpus selected from the **94th Legislature**.

---

## Current Status

Implemented:

- PDF text extraction
- legislative-text cleaning and chunking
- one-summary-per-bill summarization pipeline
- Chroma-based indexing layer
- baseline RAG pipeline
- summary-augmented RAG pipeline
- manual retrieval smoke tests
- first-pass evaluation workflow and question set

Early result:

- the **summary-augmented system** performed slightly better overall on the first evaluation pass, especially for **broad, cross-bill questions**
- the **baseline system** remained strong on **exact, single-bill detail questions**

---

## Tech Stack

- Python
- OpenAI API
- LangChain
- ChromaDB
- PyPDF
- pytest

---

## Repository Structure

```text
legislation-rag/
├── README.md
├── .gitignore
├── pyproject.toml
├── .env.example
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
├── scripts/
│   ├── extract_text.py
│   ├── clean_text.py
│   ├── chunk_documents.py
│   ├── generate_summaries.py
│   ├── build_indexes.py
│   ├── smoke_test_retrieval.py
│   ├── run_baseline_rag.py
│   ├── run_summary_rag.py
│   └── evaluate_systems.py
├── src/
│   └── legislation_rag/
│       ├── __init__.py
│       ├── config.py
│       ├── ingestion/
│       ├── summarization/
│       ├── retrieval/
│       ├── rag/
│       └── utils/
├── tests/
└── docs/
```

## Setup
### 1. Cone the repository
``` bash
git clone <your-repo-url>
cd legislation-rag
``` 

### 2. Create and activate a virtual environment
Mac/Linus:
``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:
``` bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install the project
``` bash
pip install --upgrade pip
pip install -e ".[dev]"
```

### 4. Create your environment file
Mac/Linux:
``` bash
cp .env.example .env
```

Windows:
``` bash
copy .env.example .env
```

### 5. Update `.env`
Example
``` env
OPENAI_API_KEY=your_api_key_here
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4.1-mini
VECTOR_DB_DIR=./vectorstore
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed
LOG_LEVEL=INFO
```

### 6. Add source documents
Place raw bill PDFs in `data /raw/pdfs/`
Place metadatafiles in `data/raw/metadata/`

## End-to-End Pipeline
The main workflow is:

1. Extract text from PDFs
2. Clean extracted text
3. Chunk cleaned bills
4. Generate plain-English summaries
5. Build vector indexes
6. Run baseline RAG
7. Run summary-augmented RAG
8. Evaluate both systems

## Run the Main Script
### 1. Extract text from bill PDFs
runs extraction over all PDFs in `data/raw/pdfs/` and saves `.txt` files to `data/processed/text/`
``` bash
python scripts/extract_text.py
```
Optional, run on a single bill:
``` bash
python scripts/extract_text.py --file filename.pdf
```

### 2. Clean extracted bill text
Cleans PDF artifacts such as page markers, line numbers, repeated headers/footers, and invisible characters.
``` bash
python scripts/clean_text.py
```
Optional, run on a single extracted text file
``` bash
python scripts/clean_text.py --file filename.txt
```

### 3. Chunk cleaned bill text
Splits cleaned bills into overlapping retrieval-ready chunks and saves JSON chunk files.
``` bash
python scripts/chunk_documents.py
```
Optional, run on a single cleaned text file
``` bash
python scripts/chunk_documents.py --file filename.txt
```

### 4. Generate bill summaries
Generate one plain-English summary per celaned bill and saves summary JSON files.
``` bash
python scripts/generate_summaries.py
```
Optional, run on a single cleaned bill
``` bash
python scripts/generate_summaries.py --file filename.txt
```

### 5. Build vector indexes
Build two Chroma collections:
- `bill_chunks`
- `bill_chunks_plus_summaries`
``` bash
python scripts/build_indexes.py --reset
```

### 6. Run a retrieval smoke test
Useful for sanity-checking whether retrieval is returning relevant results from a collection
``` bash
python scripts/smoke_test_retrieval.py --query "Which bill changes landlord responsibilities, and how?" --collection bill_chunks_plus_summaries --k 5
```

### 7. Run baseline RAG
Queries the baseline system using original bill chunks only
``` bash
python scripts/run_baseline_rag.py --question "Which bill changes landlord responsibilities, and how?" --show-context
```

Optional: restrict retrieval to a single bill
``` bash
python scripts/run_baseline_rag.py --question "What landlord duties does this bill require?" --bill-id filename --show-context
```

### 8. Run summary-augmented RAG
Queries the augmented system using original bill chunks plus generated summaries.
``` bash
python scripts/run_summary_rag.py --question "Which bill changes landlord responsibilities, and how?" --show-context
```
Optional, restricy retrieval to a single bill:
``` bash
python scripts/run_summary_rag.py --question "What landlord duties does this bill require?" --bill-id filename --show-context
```

### 9. Run evaluation
Runs the saved evaluation question set through both systems and stores the outputs in `data/evaluation/results/`.
``` bash
python scripts/evaluate_systems.py --run-label first_pass
```

Optional, run a subset
``` bash
python scripts/evaluate_systems.py --category corpus_wide --run-label corpus_only
```

## Evaluation Design
The project currently uses a mixed evaluation set with three kinds of questions:

- Corpus-wide questions: Broad questions that require synthesis across multiple bills
- Specific-bill questions asked naturally: Questions that refer to a bill by topic rather than by bill ID
- Exact bill control questions: Focused questions tied to a known bill for tighter grounding checks

This design helps compare where each system performs better:

- summary-augmented RAG tends to perform better on broad and cross-document synthesis
- baseline RAG remains competitive on single-bill detail and procedural precision

## Summary Style
Generated bill summaries are designed to be:

- plain English
- grounded only in the bill text
- focused on what the bill changes
- free of legal advice
- written as one paragraph with enough detail to capture the main changes

## Testings
Run the preprocessing smoke test with:
``` bash
pytest tests/test_preprocessing_smoke.py -q
```

## Example Use Case
A user asks:

`Which bill changes landlord responsibilities, and how?`

The project can answer that question in two different ways:

- by retrieving only from raw bill chunks
- by retrieving from raw bill chunks plus plain-English summaries

This makes it possible to compare whether summaries improve retrieval and downstream answer quality.

## Future Improvements
Planned or possible next steps:

- weighted or mixed retrieval strategies for chunks and summaries
- better handling of generic opening bill text
- automated scoring and result aggregation
- side-by-side answer comparison UI
- hosted demo application

## Why This Project Matters
Legislative documents are important, but often inaccessible to non-experts. This project explores a practical way to make complex public-policy text easier to search, understand, and compare while still staying grounded in the source material.