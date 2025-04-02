import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor


def knn(x: Tensor, k: int) -> Tensor:
    """K-Nearest Neighbors using pairwise distance.

    Args:
        x: Input tensor with shape (B, N, D).
        k: Number of nearest neighbors.

    Returns:
        indices: Indices of k-nearest neighbors with shape (B, N, K).
    """
    pairwise_dist = torch.cdist(x, x, p=2)  # (B, N, N)
    indices = pairwise_dist.topk(k, dim=-1, largest=False)[1]  # (B, N, K)
    return indices


def get_graph_feature(x: Tensor, indices: Tensor) -> Tensor:
    """Select features from neighbors.

    Args:
        x: the input features with shape (B, N, D).
        indices: the indices indicating the K neighbors for each input point
            with shape (B, N, K).

    Returns:
        The selected features with shape (B, N, K, 2*D).
    """
    batch_size, num_points, _ = x.shape
    k = indices.shape[-1]

    # Gather neighbor features
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    indices = indices + idx_base
    indices = indices.view(-1)

    x = x.view(batch_size * num_points, -1)
    features = x[indices, :].view(batch_size, num_points, k, -1)

    # Concatenate features with central points
    x = x.view(batch_size, num_points, 1, -1).expand(-1, -1, k, -1)
    features = torch.cat((features - x, x), dim=-1)

    return features


class Dgcnn(nn.Module):
    def __init__(
        self,
        size_latent: int,
        k: int = 20,
        aggregate_ops_local: str = "max",
        aggregate_ops_global: str = "max",
    ):
        super().__init__()

        self.k = k
        self.size_latent = size_latent
        self.aggreate_ops_local = aggregate_ops_local
        self.aggreate_ops_global = aggregate_ops_global

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(128)
        self.bn_4 = nn.BatchNorm1d(256)
        self.bn_5 = nn.BatchNorm1d(self.size_latent)

        self.slope = 0.2
        self.conv_1 = nn.Linear(3 * 2, 64, bias=False)
        self.conv_2 = nn.Linear(64 * 2, 64, bias=False)
        self.conv_3 = nn.Linear(64 * 2, 128, bias=False)
        self.conv_4 = nn.Linear(128 * 2, 256, bias=False)
        self.conv_5 = nn.Linear(512, self.size_latent, bias=False)

    def block_forward(
        self,
        features: Tensor,
        conv: nn.Linear,
        bn: nn.BatchNorm1d,
        indices: Tensor,
        agggreate_ops: str,
    ) -> Tensor:
        x = get_graph_feature(features, indices)
        x = conv(x)
        x = rearrange(x, "b n k d -> b d (n k)")
        x = bn(x)
        x = F.leaky_relu(x, negative_slope=self.slope)
        features_out = rearrange(x, "b d (n k) -> b n d k", k=self.k)

        if agggreate_ops == "max":
            features_out = features_out.max(dim=-1)[0]
        elif agggreate_ops == "avg":
            features_out = features_out.mean(dim=-1)

        return features_out

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: the input point cloud with shape (B, N, 3).

        Returns:
           The global embeddings with shape (B, size_latent).
        """
        indices = knn(x, self.k)

        x1 = self.block_forward(
            x, self.conv_1, self.bn_1, indices, self.aggreate_ops_local
        )
        x2 = self.block_forward(
            x1, self.conv_2, self.bn_2, indices, self.aggreate_ops_local
        )
        x3 = self.block_forward(
            x2, self.conv_3, self.bn_3, indices, self.aggreate_ops_local
        )
        x4 = self.block_forward(
            x3, self.conv_4, self.bn_4, indices, self.aggreate_ops_local
        )
        x5 = self.conv_5(torch.cat((x1, x2, x3, x4), dim=-1))
        x5 = rearrange(x5, "b n d -> b d n")
        x5 = self.bn_5(x5)
        feat = F.leaky_relu(x5, negative_slope=self.slope)

        if self.aggreate_ops_global == "max":
            feat = feat.max(dim=-1)[0]
        elif self.aggreate_ops_global == "avg":
            feat = feat.mean(dim=-1)

        return feat
