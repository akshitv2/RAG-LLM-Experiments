from typing import List, Dict

import spacy
from datasets import load_dataset
from tqdm import tqdm
import json

from Models.Encoder.ClipTextEncoder import ClipTextEncoder
from RAGUtils.FaissUtils import FaissIndex
from config.Config import WikiPediaConfig
from utils.Chunking import chunk_record
from utils.CreateDirs import ensure_dir

def embed_corpus(encoder, text_chunks, cfg:dict):
    texts = [c["text"] for c in text_chunks]
    embeds = encoder.encode(texts, batch_size=cfg["encoder"]["batch_size"])
    dim = embeds.size(1)
    print(f"Embeddings shape: {tuple(embeds.shape)}")
    fidx = FaissIndex(dim)
    fidx.add(embeds)

        # 5) Save index & metadata
    print("Saving index and metadata...")
    fidx.save(cfg["faiss"]["index_path"])
    with open(cfg["faiss"]["meta_path"], "w", encoding="utf-8") as f:
        for c in text_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return fidx, text_chunks

def load_and_chunk_wiki(cfg, nlp):
    ensure_dir(cfg["faiss"]["out_dir"])

        # 1) Load a small slice of Wikipedia
    # print(f"Loading wikimedia/wikipedia dataset: config={cfg["wiki"]["wiki_config"]}, split={cfg["wiki"]["wiki_split"]}")
    ds = load_dataset("wikimedia/wikipedia", cfg["wiki"]["wiki_config"], split=cfg["wiki"]["wiki_split"], cache_dir = "G:/MLCache")

    if cfg["wiki"]["max_articles"] is not None:
        n = min(len(ds), cfg["wiki"]["max_articles"])
        ds = ds.select(range(n))
        print(f"Using first {n} articles from the split.")

        # 2) Chunk articles by word count
        print("Chunking articles...")
        all_chunks: List[Dict] = []
        for rec in tqdm(ds, desc="Chunking"):
            title = rec.get("title", "") or "Untitled"
            text = rec.get("text", "") or ""
            pid = rec.get("id", rec.get("pageid", ""))
            chunks = chunk_record(title, pid, text, cfg["chunking"]["max_words_per_chunk"], cfg["chunking"]["min_words_per_chunk"], nlp=nlp)
            all_chunks.extend(chunks)
        if not all_chunks:
            raise RuntimeError("No chunks produced. Try adjusting min/max words or dataset slice.")

        print(f"Total chunks: {len(all_chunks)}")
        return all_chunks
