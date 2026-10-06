from pathlib import Path

import torch
from torchvision import datasets, transforms


def main():
    # 数据存到用户缓存目录，避免OneDrive同步整个数据集
    data_root = (
        Path.home()
        / ".cache"
        / "tokenscope-vit"
    )

    # 创建一个Transform对象，此时还没有处理图片
    transform = transforms.ToTensor()

    # 创建Dataset对象
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=transform,
        download=True,
    )

    # 调用Dataset的__len__
    print("Dataset length:", len(train_dataset))

    # 相当于调用train_dataset.__getitem__(0)
    image, label = train_dataset[0]

    print("Image type:", type(image))
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Image device:", image.device)
    print("Image min:", image.min().item())
    print("Image max:", image.max().item())

    print("Label type:", type(label))
    print("Label value:", label)
    print("Class name:", train_dataset.classes[label])

    print("Class names:")
    print(train_dataset.classes)

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert 0 <= label < 10

    print("CIFAR-10 sample check passed.")


if __name__ == "__main__":
    main()