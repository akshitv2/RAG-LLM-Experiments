from __future__ import annotations

import spacy
from typing import List, Dict

def tokenize_and_clean(nlp:spacy.lang.en.English, text:str):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.like_num
        and token.is_alpha
    ]
    return tokens

def split_by_words(text: str, max_words: int,nlp:spacy.lang.en.English) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    if nlp is not None:
        words = tokenize_and_clean(nlp, text)
    for i in range(0, len(words), max_words):
        chunk_words = words[i : i + max_words]
        chunks.append(" ".join(chunk_words))
    return chunks


def chunk_record(title: str, page_id: str | int, text: str, max_words: int, min_words: int,nlp:spacy.lang.en.English) -> List[Dict]:
    chunks = []
    for j, chunk in enumerate(split_by_words(text, max_words=max_words, nlp=nlp)):
        if len(chunk.split()) < min_words:
            continue
        chunks.append({
            "title": title,
            "page_id": str(page_id),
            "chunk_id": j,
            "text": chunk,
        })
    return chunks