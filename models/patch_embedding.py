import torch
from torch import nn

from .patchify import patchify


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 4, embed_dim: int = 20):
        super().__init__()

        self.patch_size = patch_size
        self.proj = nn.Linear(
            3 * patch_size * patch_size,
            embed_dim,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = patchify(images, self.patch_size)
        tokens = self.proj(patches)
        return tokens


if __name__ == "__main__":
    model = PatchEmbedding()

    images = torch.randn(2, 3, 32, 32)
    tokens = model(images)

    print("图片 shape:", images.shape)
    print("投影层权重 shape:", model.proj.weight.shape)
    print("输出 Token shape:", tokens.shape)