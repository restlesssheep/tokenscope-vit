from torch import Tensor


def patchify(images: Tensor, patch_size: int) -> Tensor:
    batch_size, channels, height, width = images.shape

    patch_grid = images.unfold(
        2, patch_size, patch_size
    ).unfold(
        3, patch_size, patch_size
    )

    ordered_patches = patch_grid.permute(0, 2, 3, 1, 4, 5)

    patch_vectors = ordered_patches.reshape(
        batch_size,
        -1,
        channels * patch_size * patch_size,
    )

    return patch_vectors