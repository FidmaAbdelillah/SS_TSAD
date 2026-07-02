import random
import torch


def jitter(
    x: torch.Tensor,
    sigma: float = 0.02
) -> torch.Tensor:
    """
    Add Gaussian noise.
    """
    return x + sigma * torch.randn_like(x)


def scaling(
    x: torch.Tensor,
    sigma: float = 0.1
) -> torch.Tensor:
    """
    Random amplitude scaling.
    """

    scale_factor = (
        1.0 +
        sigma * torch.randn(
            x.size(0),
            1,
            device=x.device
        )
    )

    return x * scale_factor


def time_mask(
    x: torch.Tensor,
    mask_ratio: float = 0.1
) -> torch.Tensor:
    """
    Randomly mask a contiguous portion
    of each signal.
    """

    batch_size, signal_length = x.shape

    mask_length = int(signal_length * mask_ratio)

    if mask_length <= 0:
        return x

    output = x.clone()

    starts = torch.randint(
        0,
        signal_length - mask_length + 1,
        (batch_size,),
        device=x.device
    )

    for i in range(batch_size):

        start = starts[i].item()

        output[i, start:start + mask_length] = 0.0

    return output


def weak_view(x: torch.Tensor) -> torch.Tensor:
    """
    Weak augmentation branch.
    """

    x = scaling(x, sigma=0.03)
    x = jitter(x, sigma=0.01)

    return x


def strong_view(x: torch.Tensor) -> torch.Tensor:
    """
    Strong augmentation branch.
    """

    x = scaling(x, sigma=0.09)
    x = jitter(x, sigma=0.03)

    # Optional:
    # x = time_mask(x, mask_ratio=0.12)

    return x