from dataclasses import dataclass

@dataclass
class WikiPediaConfig:
    # Wikipedia
    wiki_config: str = "20231101.en"   # wikimedia/wikipedia config (language+dump date)
    wiki_split: str = "train[:1%]"     # small slice for demo
    max_articles: int | None = 100000

    # Chunking
    max_words_per_chunk: int = 128
    min_words_per_chunk: int = 40

    # Embedding
    clip_model_name: str = "openai/clip-vit-large-patch14"
    # clip_model_name: str = "openai/clip-vit-base-patch32"
    batch_size: int = 64

    # FAISS / persistence
    out_dir: str = "./faiss_db"
    index_path: str = "./faiss_db/index.faiss"
    meta_path: str = "./faiss_db/metadata.jsonl"

    # LLM
    llm_model_name: str = "google/flan-t5-base"  # demo model
    llm_max_new_tokens: int = 200
