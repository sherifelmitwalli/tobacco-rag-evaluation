# RAG Pipeline for Document Analysis

A sophisticated Retrieval-Augmented Generation (RAG) pipeline for analyzing documents through three distinct phases of processing, model comparison, and optimization.

## Overview

This project implements a comprehensive RAG pipeline that processes documents, generates queries, compares language model performances, and optimizes retrieval parameters. The system is divided into three phases:

### Phase 1: Data Collection and Query Generation
- Processes PDF documents and splits them into manageable chunks
- Utilizes ChromaDB for efficient document storage and retrieval
- Implements custom embedding using SentenceTransformer
- Generates topic-specific queries covering:
  - Industry strategies/practices
  - Business/market performance
  - Tobacco/nicotine products
  - Science/health effects
  - Policy/regulation
- Uses GPT-4 for response generation and evaluation
- Outputs query-response pairs to CSV

### Phase 2: Model Comparison
- Evaluates two language models:
  - Mixtral-8x7B
  - Llama-3-70B
- Performs comprehensive evaluation using 9 criteria:
  - Relevance
  - Accuracy
  - Completeness
  - Clarity
  - Conciseness
  - Coherence
  - Context Awareness
  - Adaptability
  - Domain Specificity
- Includes statistical analysis and visualization of model performance
- Generates radar charts for model comparison

### Phase 3: Hyperparameter Optimization
- Implements grid search for parameter optimization:
  - Chunk sizes: [100, 300, 500]
  - Overlaps: [0%, 25%, 50%]
  - Embedding models: ['all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2']
  - Retrieved passages: [1, 3, 5]
- Produces visualizations of grid search results
- Generates final results using optimal parameters

## Requirements

```
pypdf
openai==0.28
chromadb
typing
sentence-transformers
transformers
deepinfra[openai]
numpy
pandas
matplotlib
torch
```

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up API keys:
- OpenAI API key for GPT-4 evaluations
- DeepInfra API key for model inference. You'll need access to:
  - `mistralai/Mixtral-8x7B-Instruct-v0.1`
  - `meta-llama/Meta-Llama-3.1-70B-Instruct`
  Visit [DeepInfra](https://deepinfra.com/) to obtain API access for these models

3. Prepare your input:
- Place your PDF document in the project directory
- Update the file path in each phase's notebook

## Usage

The pipeline is implemented in three Jupyter notebooks:

1. Run `Phase1.ipynb` to process the document and generate queries
2. Run `Phase2.ipynb` to compare model performances
3. Run `Phase3.ipynb` to optimize parameters and generate final results

## Output

The pipeline generates several output files:

- `query_response_pairs.csv`: Initial query-response pairs
- `mixtral_responses.csv`: Responses from Mixtral model
- `llama_responses.csv`: Responses from Llama model
- `response_evaluations.csv`: Evaluation scores
- `grid_search_results.csv`: Parameter optimization results
- `final_results.csv`: Final results with optimal parameters
- Various visualization plots showing model performance and parameter optimization results
