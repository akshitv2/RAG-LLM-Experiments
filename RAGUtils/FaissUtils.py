import faiss
from typing import List, Tuple

import faiss
import torch


class FaissIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, xb: torch.Tensor):
        if xb.dtype != torch.float32:
            xb = xb.float()
        self.index.add(xb.numpy())

    def search(self, xq: torch.Tensor, top_k: int = 5) -> Tuple[List[List[int]], List[List[float]]]:
        if xq.ndim == 1:
            xq = xq.unsqueeze(0)
        if xq.dtype != torch.float32:
            xq = xq.float()
        scores, idx = self.index.search(xq.numpy(), top_k)
        return idx.tolist(), scores.tolist()

    def save(self, path: str):
        faiss.write_index(self.index, path)

    @staticmethod
    def load(path: str) -> "FaissIndex":
        index = faiss.read_index(path)
        obj = FaissIndex(index.d)
        obj.index = index
        return obj