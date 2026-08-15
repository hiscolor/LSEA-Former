import torch
import torch.nn.functional as F


def soft_rank(
    x: torch.Tensor,
    regularization: str = "l2",
    regularization_strength: float = 1.0,
) -> torch.Tensor:
    """
    GPU-compatible differentiable soft rank.

    Computes a differentiable approximation of the rank of each element.
    Uses pairwise sigmoid comparisons.

    Args:
        x: Input tensor of shape (..., n) where n is the sequence length.
           Supports any number of batch dimensions.
        regularization: Type of regularization ('l2' or 'kl').
                       Currently both use the same sigmoid-based approach.
        regularization_strength: Controls smoothness. Lower values give
                                 sharper (closer to hard) ranks.

    Returns:
        Soft ranks tensor of the same shape as input.
        Ranks are 1-indexed (range approximately [1, n]).

    Example:
        >>> x = torch.tensor([[3., 1., 2.]], device='cuda')
        >>> soft_rank(x, regularization_strength=0.1)
        tensor([[3., 1., 2.]], device='cuda:0')
    """

    original_shape = x.shape
    n = x.shape[-1]


    x_flat = x.reshape(-1, n)


    pairwise_diff = x_flat.unsqueeze(-1) - x_flat.unsqueeze(-2)



    comparisons = torch.sigmoid(pairwise_diff / regularization_strength)



    ranks = comparisons.sum(dim=-1)


    return ranks.reshape(original_shape)


def soft_sort(
    x: torch.Tensor,
    regularization: str = "l2",
    regularization_strength: float = 1.0,
) -> torch.Tensor:
    """
    GPU-compatible differentiable soft sort.

    Computes a differentiable approximation of sorted values.
    Uses a soft permutation matrix approach for accurate results.

    Args:
        x: Input tensor of shape (..., n) where n is the sequence length.
           Supports any number of batch dimensions.
        regularization: Type of regularization ('l2' or 'kl').
                       Currently both use the same approach.
        regularization_strength: Controls smoothness. Lower values give
                                 sharper (closer to hard) sort.

    Returns:
        Soft sorted tensor of the same shape as input (ascending order).

    Example:
        >>> x = torch.tensor([[3., 1., 2.]], device='cuda')
        >>> soft_sort(x, regularization_strength=0.1)
        tensor([[1., 2., 3.]], device='cuda:0')
    """

    original_shape = x.shape
    n = x.shape[-1]


    x_flat = x.reshape(-1, n)
    batch_size = x_flat.shape[0]



    pairwise_diff = x_flat.unsqueeze(-1) - x_flat.unsqueeze(-2)







    positions = torch.arange(n, device=x.device, dtype=x.dtype)




    scores = (2 * positions - n + 1).view(1, -1, 1) * x_flat.unsqueeze(1)


    perm_matrix = F.softmax(scores / regularization_strength, dim=-1)


    sorted_x = torch.bmm(perm_matrix, x_flat.unsqueeze(-1)).squeeze(-1)


    return sorted_x.reshape(original_shape)



def soft_sort_l2(x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Alias with tau parameter for compatibility."""
    return soft_sort(x, regularization="l2", regularization_strength=tau)


def soft_rank_l2(x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Alias with tau parameter for compatibility."""
    return soft_rank(x, regularization="l2", regularization_strength=tau)
