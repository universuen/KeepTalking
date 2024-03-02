import torch
from torch import nn
import torchvision.transforms as transforms


class LearnableVisualPrompts(nn.Module):
    def __init__(self, num_channels: int, height: int, width: int) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(
            torch.randn(1, num_channels, height, width)
        )

    @torch.no_grad()
    def normalize(self) -> None:
        normalized_embeddings = (self.embeddings - self.embeddings.min()) / (self.embeddings.max() - self.embeddings.min())
        self.embeddings.data = normalized_embeddings
        
    def to_image(self):
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(self.embeddings.squeeze(0).cpu().detach())

        return pil_image

