from typing import List, Dict

import torch
from transformers import pipeline

from Models.Encoder.ClipTextEncoder import ClipTextEncoder
from config.Config import WikiPediaConfig

from RAGUtils.FaissUtils import FaissIndex

def search_query(query: str, fidx: FaissIndex, encoder: ClipTextEncoder, metadata: List[Dict], top_k: int = 5):
    q_emb = encoder.encode([query])  # already normalized
    idxs, scores = fidx.search(q_emb, top_k=top_k)
    idxs = idxs[0]
    scores = scores[0]

    print("\nTop matches:")
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        rec = metadata[i]
        print(f"{rank:>2}. score={s:.3f} | title={rec['title']} (page_id={rec['page_id']}, chunk={rec['chunk_id']})")
        preview = rec["text"].replace("\n", " ")
        print("    " + (preview[:180] + ("..." if len(preview) > 180 else "")))

def search_and_answer(query: str, fidx: FaissIndex, encoder: ClipTextEncoder, metadata: List[Dict], cfg: dict, top_k: int = 5):
    q_emb = encoder.encode([query])
    idxs, scores = fidx.search(q_emb, top_k=top_k)
    idxs = idxs[0]
    scores = scores[0]

    print("\nTop retrieved matches:")
    retrieved_chunks = []
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        rec = metadata[i]
        retrieved_chunks.append(rec["text"])
        print(f"{rank:>2}. score={s:.3f} | title={rec['title']} (page_id={rec['page_id']}, chunk={rec['chunk_id']})")

    # Build context for LLM
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Answer the question using only the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    print("\nSending to LLM...\n")
    llm = pipeline("text2text-generation", model=cfg["llm"]["llm_model_name"], device=0 if torch.cuda.is_available() else -1)
    reply = llm(prompt, max_new_tokens=cfg["llm"]["llm_max_new_tokens"], do_sample=False)[0]["generated_text"]

    print("LLM Reply:\n" + reply.strip())