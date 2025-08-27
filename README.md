# RAG-LLM-Experiments
### Wikipedia Dataset used for RAG with FAISS, CLIP, spaCy, and Hugging Face LLM

This project creates a **Retrieval-Augmented Generation (RAG)** pipeline:
1. Load and clean **Wikipedia text** (from Hugging Face Datasets).
2. Split into chunks, clean/tokenize with **spaCy**.
3. Encode with **CLIP text encoder** and store embeddings in a **FAISS vector database**.
4. At query time, search FAISS for the most relevant chunks.
5. Feed retrieved passages + the query into a **Hugging Face LLM** (Flan-T5) to generate an answer.

---

## 🔧 Requirements

- Python 3.9+
- PyTorch
- Hugging Face `transformers` + `datasets`
- FAISS
- spaCy

Install dependencies:
```bash
pip install torch faiss-cpu transformers datasets spacy
```

## Configuration
Inside Config.py you can adjust:
- **NUM_ARTICLES**: how many Wikipedia entries to embed
- **CHUNK_SIZE**: words per chunk
- **CLIP_MODEL**: CLIP variant (e.g., openai/clip-vit-base-patch32)
- **LLM_MODEL**: Hugging Face model for answering (e.g., google/flan-t5-base)
- **TOP_K**: number of passages to retrieve per query

Sources:
- Faiss Implementation: https://qxf2.com/blog/build-semantic-search-faiss/
- NLP : https://www.nlplanet.org/course-practical-nlp/02-practical-nlp-first-tasks/11-multilingual-search-recsys-wikipedia