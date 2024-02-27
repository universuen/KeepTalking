import torch
from torch import nn


class LearnablePrompts(nn.Module):
    def __init__(self, num_prompts: int, num_dims: int) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(
            torch.randn(num_prompts, num_dims)
        )

    def to_ids(self, embedding_layer: torch.nn.Embedding, batch_size: int = 10) -> torch.Tensor:
        all_ids = []
        for i in range(0, self.embeddings.size(0), batch_size):
            batch_embeddings = self.embeddings[i:i+batch_size]
            cosine_similarities = torch.cosine_similarity(
                batch_embeddings.unsqueeze(1), 
                embedding_layer.weight.data.unsqueeze(0), 
                dim=-1
            )
            ids = torch.argmax(cosine_similarities, dim=-1)
            all_ids.append(ids)
        all_ids = torch.cat(all_ids, dim=0)
        return all_ids

