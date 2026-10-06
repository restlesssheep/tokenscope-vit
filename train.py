from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.linear_classifier import LinearClassifier


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)

    data_root = Path.home() / ".cache" / "tokenscope-vit"

    # 训练集：50,000 张图片
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=transforms.ToTensor(),
        download=False,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    # 测试集：10,000 张训练时没见过的图片
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=transforms.ToTensor(),
        download=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    model = LinearClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    # ---------- 训练一轮 ----------
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        predictions = logits.argmax(dim=1)
        correct = (predictions == labels).sum().item()

        train_loss_sum += loss.item() * batch_size
        train_correct += correct
        train_samples += batch_size

    print(
        f"训练集：平均 loss={train_loss_sum / train_samples:.4f}, "
        f"正确率={train_correct}/{train_samples} "
        f"({train_correct / train_samples:.1%})"
    )

    # ---------- 测试 ----------
    model.eval()

    test_loss_sum = 0.0
    test_correct = 0
    test_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            predictions = logits.argmax(dim=1)
            correct = (predictions == labels).sum().item()

            test_loss_sum += loss.item() * batch_size
            test_correct += correct
            test_samples += batch_size

    print(
        f"测试集：平均 loss={test_loss_sum / test_samples:.4f}, "
        f"正确率={test_correct}/{test_samples} "
        f"({test_correct / test_samples:.1%})"
    )
    checkpoint_path = data_root / "linear_cifar10.pt"

    torch.save(model.state_dict(), checkpoint_path)

    print("权重保存位置：", checkpoint_path)
    print("文件存在：", checkpoint_path.is_file())


if __name__ == "__main__":
    main()