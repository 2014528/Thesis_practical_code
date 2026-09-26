# Thesis Practical Code

This repository contains the practical implementation and experimental code for my master's thesis on **Retrieval-Augmented Generation (RAG) for question answering over university academic regulations**.

The study investigates whether **retrieval strategy** and **document chunking configuration** affect retrieval performance and downstream answer quality.

## Research Focus

The practical implementation compares three retrieval approaches:

1. **BM25 retrieval**
2. **Dense retrieval using `all-MiniLM-L6-v2`**
3. **Hybrid retrieval combining BM25 and dense retrieval**

Each retrieval approach is evaluated using three document chunking configurations:

* Small chunks
* Medium chunks
* Large chunks

This produces **9 primary retrieval × chunking experimental conditions**.

The study evaluates retrieval performance and then uses the selected retrieval configuration in a RAG pipeline for answer generation.

---

## Models Used

### Dense Retrieval / Embeddings

**`sentence-transformers/all-MiniLM-L6-v2`**

This model is used to create embeddings for the document chunks and user questions. Cosine-style similarity is then used to identify semantically relevant chunks.

### Answer Generation

**`google/flan-t5-base`**

The retrieved contexts are provided to FLAN-T5-base to generate answers to questions about university academic regulations.

The retrieval model and generation model therefore have different roles:

* `all-MiniLM-L6-v2` → retrieves semantically relevant information
* `google/flan-t5-base` → generates the final answer

---

## Retrieval Methods

### BM25

BM25 is used as the lexical retrieval method. It ranks document chunks based on the relationship between the words in the question and the words in each document chunk.

### Dense Retrieval

Dense retrieval represents questions and document chunks as embeddings using `all-MiniLM-L6-v2`. The system ranks chunks according to their semantic similarity to the question.

### Hybrid Retrieval

Hybrid retrieval combines BM25 and dense retrieval scores.

Before combining the scores, both retrieval scores are normalised using min-max normalisation.

The hybrid score is calculated as:

`Hybrid Score = α × Normalised BM25 + (1 − α) × Normalised Dense Score`

The implementation supports different α values, allowing the relative contribution of BM25 and dense retrieval to be investigated.

---

## Evaluation Metrics

Retrieval performance is evaluated at:

* **k = 1**
* **k = 3**
* **k = 5**

The following retrieval metrics are calculated:

* Precision@k
* Recall@k
* Mean Reciprocal Rank (MRR@k)
* Normalised Discounted Cumulative Gain (nDCG@k)

The implementation evaluates relevance primarily at the document level and can also evaluate supporting chunk relevance when a supporting chunk ID is available.

---

## RAG Answer Generation

After the retrieval experiments, the selected retrieval configuration is used in the RAG answer-generation stage.

The system:

1. Receives a question.
2. Retrieves relevant document chunks.
3. Selects the highest-ranked contexts.
4. Places the retrieved contexts into a structured prompt.
5. Sends the prompt to `google/flan-t5-base`.
6. Generates an answer using only the provided context.

The prompt instructs the model to return:

> "Not found in the provided context."

when the requested information is not available in the retrieved context.

---

## Answer Quality Evaluation

The generated answers are evaluated using semantic and context-support measures.

The implementation uses `all-MiniLM-L6-v2` to calculate semantic similarity between generated answers and reference answers.

It also calculates a context-support score by comparing answer sentences with the retrieved context.

These automated measures are used as evaluation measures/proxies for downstream answer quality and contextual support.

---

## Dataset Structure

The notebook expects a dataset ZIP containing processed data with the required files and fields.

The implementation expects:

### Question-answer data

Required columns include:

* `question_id`
* `question`
* `reference_answer`
* `doc_id`

### Chunk data

Required columns include:

* `chunk_id`
* `doc_id`
* `chunk_text`

The implementation supports separate datasets for:

* small chunks
* medium chunks
* large chunks

If `supporting_chunk_id` is available in the question-answer dataset, it is also used for chunk-level relevance evaluation.

---

## Experimental Workflow

The overall workflow implemented in the notebook is:

```text
University Regulation Documents
            ↓
       Preprocessing
            ↓
        Document Chunking
            ↓
 ┌──────────┼──────────┐
 ↓          ↓          ↓
BM25      Dense      Hybrid
            ↓
       Retrieval Ranking
            ↓
   Retrieval Evaluation
            ↓
  Select Best Configuration
            ↓
      Top Retrieved Context
            ↓
      FLAN-T5-base
            ↓
     Generated Answer
            ↓
    Answer Quality Evaluation
            ↓
       Tables & Figures
```

---

## Repository Contents

The main practical implementation is provided as a Jupyter Notebook:

`Thesis_practical_code.ipynb`

The notebook contains the complete workflow for:

* installing and importing required libraries
* uploading and extracting the dataset
* validating and cleaning the data
* implementing BM25 retrieval
* implementing dense retrieval
* implementing hybrid retrieval
* running retrieval experiments
* calculating evaluation metrics
* selecting the best retrieval configuration
* generating RAG answers
* evaluating answer quality
* creating Chapter 4 tables
* generating visualisations
* exporting the final results

---

## Required Python Libraries

The notebook installs the main dependencies required for the implementation:

```text
pandas
numpy
scikit-learn
rank-bm25
sentence-transformers
transformers
accelerate
torch
matplotlib
tqdm
```

The notebook is designed to run in a **Google Colab** environment and uses Colab's file-upload functionality for the dataset.

---

## Running the Notebook

### 1. Open the notebook

Open `Thesis_practical_code.ipynb` in Google Colab or another compatible Jupyter environment.

### 2. Install the dependencies

Run the first code cell to install the required Python packages.

### 3. Upload the dataset

The notebook prompts the user to upload the dataset ZIP file.

The ZIP file should contain the expected processed dataset structure, including the metadata and chunk/question-answer files.

### 4. Run the notebook cells

Run the cells in order.

The notebook will:

* load and validate the dataset
* build the retrieval systems
* run the retrieval experiments
* calculate evaluation metrics
* generate the RAG answers
* evaluate answer quality
* create tables and figures
* package the Chapter 4 results

---

## Reproducibility

The experiment uses fixed retrieval methods, chunking configurations, models and evaluation procedures so that the different retrieval configurations can be compared under controlled conditions.

The dense retrieval results specifically relate to the implementation using:

`sentence-transformers/all-MiniLM-L6-v2`

The answer-generation results specifically relate to:

`google/flan-t5-base`

Therefore, the findings should be interpreted in the context of these specific models and implementation choices.

---

## Purpose of This Repository

This repository is provided as the practical implementation accompanying the thesis.

It supports the experimental investigation of:

> **Whether retrieval strategy and document chunking configuration affect retrieval quality and downstream answer quality when answering questions over university academic regulations.**
