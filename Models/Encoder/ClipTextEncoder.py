import torch
from tqdm import tqdm
from transformers import CLIPTokenizerFast, CLIPModel
import torch.nn.functional as F
from typing import List


class ClipTextEncoder:
    def __init__(self, model_name: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """Return L2-normalized embeddings (for cosine/IP search)."""
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self.model.get_text_features(**inputs)
            feats = F.normalize(feats, p=2, dim=1)
            all_embeds.append(feats.cpu())
        return torch.cat(all_embeds, dim=0)