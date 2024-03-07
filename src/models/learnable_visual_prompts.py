import torch
from torch import nn
import torchvision.transforms as transforms

from KeepTalking.src.models.learnable_prompts import LearnablePrompts


class LearnableVisualPrompts(LearnablePrompts):
    def __init__(self, num_channels: int, height: int, width: int, seed: int = 42) -> None:
        super().__init__(1, 1)
        torch.manual_seed(seed)
        self.embeddings = nn.Parameter(
            torch.ones(1, num_channels, height, width)
        )

    @torch.no_grad()
    def normalize(self) -> None:
        normalized_embeddings = (self.embeddings - self.embeddings.min()) / (self.embeddings.max() - self.embeddings.min())
        self.embeddings.data = normalized_embeddings
        
    def to_image(self):
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(self.embeddings.squeeze(0).cpu().detach())

        return pil_image
    
    def to_ids(self) -> None:
        raise NotImplementedError
