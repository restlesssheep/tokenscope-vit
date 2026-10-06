from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    data_root = (
        Path.home()
        / ".cache"
        / "tokenscope-vit"
    )

    dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=transforms.ToTensor(),
        download=False,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    print("Dataset length:", len(dataset))
    print("DataLoader length:", len(loader))

    batch_iterator = iter(loader)

    print("Iterator type:", type(batch_iterator))

    images, labels = next(batch_iterator)

    print("Images type:", type(images))
    print("Images shape:", images.shape)
    print("Images dtype:", images.dtype)
    print("Images device:", images.device)

    print("Labels type:", type(labels))
    print("Labels shape:", labels.shape)
    print("Labels dtype:", labels.dtype)
    print("Labels:", labels)

    assert images.shape == (8, 3, 32, 32)
    assert labels.shape == (8,)

    print("DataLoader batch check passed.")


if __name__ == "__main__":
    main()
    