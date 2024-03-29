from pathlib import Path

import torch
from torch import nn


class LearnablePrompts(nn.Module):
    def __init__(self, num_prompts: int, num_dims: int, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embeddings = nn.Parameter(
            torch.randn(num_prompts, num_dims)
        )

    def to_ids(self, embedding_layer: torch.nn.Embedding, chunk_size: int = 10) -> torch.Tensor:
        all_ids = []
        for i in range(0, self.embeddings.size(0), chunk_size):
            batch_embeddings = self.embeddings[i:i + chunk_size]
            distances = torch.cdist(batch_embeddings, embedding_layer.weight.data)
            ids = torch.argmin(distances, dim=-1)
            all_ids.append(ids)
        all_ids = torch.cat(all_ids, dim=0)
        return all_ids

    def save(self, file_path: Path) -> None:
        torch.save(self.embeddings.data, file_path)

    def load(self, file_path: str) -> None:
        self.embeddings.data = torch.load(file_path)

