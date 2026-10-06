import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self):
        

        super().__init__()

        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(3 * 32 * 32, 10)

        
    def forward(self, x):
        

        x = self.flatten(x)
        logits = self.classifier(x)

        return logits


if __name__ == "__main__":
    print("1. 准备创建模型")

    model = LinearClassifier()

    print("4. 模型创建完成")

    images = torch.randn(4, 3, 32, 32)

    print("5. 准备调用 model(images)")

    logits = model(images)

    print("8. model(images) 调用完成")
    print("9. 输出 shape：", logits.shape)