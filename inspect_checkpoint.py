from pathlib import Path
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from models.patchify import patchify
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.linear_classifier import LinearClassifier


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    data_root = Path.home() / ".cache" / "tokenscope-vit"
    checkpoint_path = data_root / "linear_cifar10.pt"

    # 先创建同样结构的模型，再用保存的权重替换初始权重
    model = LinearClassifier().to(device)
    saved_weights = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(saved_weights)
    model.eval()

    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=transforms.ToTensor(),
        download=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits = model(images)

    predictions = logits.argmax(dim=1)

    print("images shape:", images.shape)
    print("logits shape:", logits.shape)
    print("labels:", labels.tolist())
    print("predictions:", predictions.tolist())
    print("本批预测正确:", (predictions == labels).sum().item(), "/8")
    patch_size = 4
    batch_size, channels, height, width = images.shape

    patch_grid = images.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )
    ordered_patches = patch_grid.permute(0, 2, 3, 1, 4, 5)
    patch_vectors = ordered_patches.reshape(
        batch_size,
        -1,
        channels * patch_size * patch_size,
    )
    function_patches = patchify(images, patch_size)
    print(
        "函数结果与旧代码完全相同:",
        torch.equal(function_patches, patch_vectors),
    )

    print("切块后 shape:", patch_grid.shape)
    print("调整顺序后 shape:", ordered_patches.shape)
    print("Patch 向量 shape:", patch_vectors.shape)
    print("元素数量不变:", images.numel() == patch_vectors.numel())
        # [B, 64, 48] → [B, 8, 8, 3, 4, 4]
    restored_grid = patch_vectors.reshape(
        batch_size,
        height // patch_size,
        width // patch_size,
        channels,
        patch_size,
        patch_size,
    )

    # [B, 8, 8, 3, 4, 4] → [B, 3, 8, 4, 8, 4] → [B, 3, 32, 32]
    restored_images = restored_grid.permute(
        0, 3, 1, 4, 2, 5
    ).reshape(batch_size, channels, height, width)

    print("还原图片 shape:", restored_images.shape)
    print("与原图完全相同:", torch.equal(restored_images, images))
    scale = 8

    # images 在 GPU 上；先取第一张并移回 CPU，才能转换成 PIL 图片
    preview = to_pil_image(images[0].cpu())

    # 32×32 → 256×256；NEAREST 保留清晰的像素边界
    preview = preview.resize(
        (width * scale, height * scale),
        resample=Image.Resampling.NEAREST,
    )

    draw = ImageDraw.Draw(preview)

    # 每隔 4 个原图像素画一条竖线和横线
    for position in range(patch_size, width, patch_size):
        line_position = position * scale

        draw.line(
            (line_position, 0, line_position, height * scale - 1),
            fill=(255, 0, 0),
            width=2,
        )
        draw.line(
            (0, line_position, width * scale - 1, line_position),
            fill=(255, 0, 0),
            width=2,
        )

    output_path = (
        Path(__file__).resolve().parent
        / "outputs"
        / "patch_grid.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path)

    print("Patch 网格图保存到:", output_path)
    embed_dim = 20

    patch_projector = torch.nn.Linear(
        channels * patch_size * patch_size,
        embed_dim,
    ).to(device)

    with torch.no_grad():
        patch_tokens = patch_projector(patch_vectors)

    print("投影层权重 shape:", patch_projector.weight.shape)
    print("投影层偏置 shape:", patch_projector.bias.shape)
    print("投影后 Token shape:", patch_tokens.shape)
    cls_token = torch.zeros(
        patch_tokens.shape[0],  # 8 张图片
        1,                      # 每张图片 1 个 CLS Token
        patch_tokens.shape[2],  # 与 Patch Token 一样是 20 维
        device=patch_tokens.device,
    )

    tokens_with_cls = torch.cat(
        (cls_token, patch_tokens),
        dim=1,
    )

    print("CLS shape:", cls_token.shape)
    print("合并后 shape:", tokens_with_cls.shape)
    print(
        "原 Patch 是否保留:",
        torch.equal(tokens_with_cls[:, 1:, :], patch_tokens),
    )
    position_ids = torch.arange(
        tokens_with_cls.shape[1],
        device=tokens_with_cls.device,
    )

    print("位置编号 shape:", position_ids.shape)
    print("前十个位置编号:", position_ids[:10].tolist())
    position_table = torch.nn.Embedding(
        num_embeddings=65,  # 可查询的位置编号：0～64
        embedding_dim=20,   # 每个位置对应 20 个数
    ).to(device)

    with torch.no_grad():
        position_vectors = position_table(position_ids)

    print("位置表权重 shape:", position_table.weight.shape)
    print("查出的所有位置向量 shape:", position_vectors.shape)
    print("编号 10 的位置向量 shape:", position_vectors[10].shape)
    position_for_batch = position_vectors.unsqueeze(0)
    tokens_with_position = tokens_with_cls + position_for_batch

    print("增加 batch 维后的 shape:", position_for_batch.shape)
    print("加上位置后的 Token shape:", tokens_with_position.shape)


if __name__ == "__main__":
    main()