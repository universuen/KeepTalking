import torch
from torch import nn


class LearnablePrompts(nn.Module):
    def __init__(self, num_prompts: int, num_dims: int) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(
            torch.randn(num_prompts, num_dims)
        )

    def to_ids(self, embedding_layer: torch.nn.Embedding) -> torch.Tensor:
        cosine_similarities = torch.cosine_similarity(
            self.embeddings.unsqueeze(1), 
            embedding_layer.weight.data.unsqueeze(0), 
            dim=-1
        )
        ids = torch.argmax(cosine_similarities, dim=-1)
        return ids

