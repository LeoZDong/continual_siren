from collections import deque
from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn

import utils


class HashEmbedding(nn.Module):
    """Multi-resolution hash function as coordinate embedding."""

    def __init__(
        self,
        coord_dim: int,
        n_levels: int,
        n_features_per_entry: int,
        log2_hashtable_size: int,
        base_resolution: int,
        finest_resolution: int,
        **kwargs,
    ) -> None:
        """Initialize a hash embedding.
        Args:
            coord_dim: Dimension of coordinate (either 2 or 3).
            n_levels: Number of levels for the multi-resolution hash
                (i.e. `L` in InstantNGP paper).
            n_features_per_entry: Number of feature dimensions for each hash table entry
                (i.e. `F` in InstantNGP paper).
            log2_hashtable_size: Log of the hash table size
                (i.e. log_2 of the `T` in InstantNGP paper).
            base_resolution: Coarsest resolution for the multi-resolution hash
                (i.e. `N_min` in InstantNGP paper).
            finest_resolution: Finest resolution for the multi-resolution hash
                (i.e. `N_max` in InstantNGP paper).
        """
        super().__init__()
        self.coord_dim = coord_dim
        self.n_levels = n_levels
        self.n_features_per_entry = n_features_per_entry
        self.log2_hashtable_size = log2_hashtable_size
        self.base_resolution = base_resolution

        # Factor of geometric progression for the resolution from coarse to fine
        self.coarse_to_fine_factor = np.exp(
            (np.log(finest_resolution) - np.log(base_resolution)) / (self.n_levels - 1)
        )

        # Inifialize embeddings for each level (i.e. lookup table)
        embeddings = []
        for _ in range(self.n_levels):
            embedding = nn.Embedding(
                2**self.log2_hashtable_size, n_features_per_entry
            )
            # Custom initialization
            nn.init.uniform_(embedding.weight, a=-1e-4, b=1e-4)
            embeddings.append(embedding)
        self.embeddings = nn.ModuleList(embeddings)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass to embed a batch of coordinates.
        Args:
            x: (bsz, 2 or 3) Batch of coordinates.
        Returns:
            (bsz, n_levels * n_features_per_entry) Multi-resolution embeddings of the
                batch of coordinates (concatenated).
        """
        x_embedding_multi_level = []
        for i in range(self.n_levels):
            # Get adjacent vertices at this level's resolution
            resolution = np.floor(
                self.base_resolution * self.coarse_to_fine_factor**i
            )
            (
                vertex_indices,
                min_vertex_coords,
                max_vertex_coords,
            ) = utils.get_adjacent_vertices(
                x, grid_min=-1, grid_max=1, vertex_resolution=resolution
            )

            # Query hash function and lookup table to get adjacent vertices' embeddings
            vertex_indices_hash = self.spatial_hash(vertex_indices, resolution)
            vertex_embeddings = self.embeddings[i](vertex_indices_hash)

            # Interpolate adjacent vertices' embeddings to get x's embedding
            x_embedding = utils.linear_interpolate(
                x, min_vertex_coords, max_vertex_coords, vertex_values=vertex_embeddings
            )
            x_embedding_multi_level.append(x_embedding)

        x_embedding_multi_level = torch.cat(x_embedding_multi_level, dim=-1)
        return x_embedding_multi_level

    def spatial_hash(self, vertex_indices: Tensor, resolution: int) -> Tensor:
        """Spatial hash function as defined in InstantNGP.
        Args:
            vertex_indices: (bsz, num_vertex, 2 or 3) Batch of vertex indices in 2D or 3D.
            resolution: Current resolution of the grid (number of vertices per side).
                Not used in this class.

        Returns: (bsz, num_vertex) Hash value for each vertex.
        """
        primes = [
            1,
            2654435761,
            805459861,
        ]

        xor_result = torch.zeros_like(vertex_indices)[..., 0]
        for i in range(vertex_indices.shape[-1]):
            xor_result ^= vertex_indices[..., i] * primes[i]

        # If N is a power of 2, then (num % N) is equivalent to (num & (N - 1)), i.e. we
        # just strip the significant digits of num (although no significant speed diff).
        return (
            torch.tensor((1 << self.log2_hashtable_size) - 1, device=xor_result.device)
            & xor_result
        )


class HashEmbeddingUnravel(HashEmbedding):
    """Hash embedding where we linearly unravel the coordinate directly as hash table
    indices. We only use the spatial hash function to compute hash indices when the
    current grid resolution has more coordinates than the hash table size.

    This is consistent with the official CUDA implementation. It is almost the same as
    always using the spatial hash function, except we strictly have no collisions when
    the grid size is less than the hash table size.

    Empirically, this performs a bit better in both continual and non-continual settings.
    """

    def __init__(self, **kwargs):
        """Initialize `self.strides` instance variable, which contains one `stride` for
        each grid `resolution`. `stride` is used to linearly unravel D-dim coordinates.
        For example, if D = 2 on a grid resolution of 16, then `stride` is [1, 16].
        Given a 2-dim coordinate of `coord = [x, y]`, its linearly unraveled index is
        `(coord * stride).sum() = (x + 16y)`.
        """

        super().__init__(**kwargs)
        # Pre-compute strides at all resolution levels
        self.strides = {}
        for i in range(self.n_levels):
            resolution = np.floor(
                self.base_resolution * self.coarse_to_fine_factor**i
            )
            stride = torch.ones([1, 1, self.coord_dim], dtype=torch.long)

            s = 1
            for dim in range(self.coord_dim):
                stride[..., dim] = s
                s *= resolution
            self.strides[resolution] = stride

    def spatial_hash(self, vertex_indices: Tensor, resolution: int) -> Tensor:
        """Spatial hash function for the unravel hash scheme. We only query the spatial
        hash function for hash indices when the current grid `resolution` has more
        coordinates than the hash table size. Otherwise, we linearly unravel the D-dim
        coordinate to get the hash indices.
        """
        primes = [
            1,
            2654435761,
            805459861,
        ]

        size = resolution**self.coord_dim
        if np.log2(size) > self.log2_hashtable_size:
            xor_result = torch.zeros_like(vertex_indices)[..., 0]
            for dim in range(self.coord_dim):
                xor_result ^= vertex_indices[..., dim] * primes[dim]
            hash_indices = (
                torch.tensor(
                    (1 << self.log2_hashtable_size) - 1, device=xor_result.device
                )
                & xor_result
            )
        else:
            stride = self.strides[resolution]
            hash_indices = (vertex_indices * stride).sum(-1)

        return hash_indices


class HashNet(nn.Module):
    """Network that uses a multi-resolution hash function as coordinate embedding."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool,
        hash_embedding: HashEmbedding,  # Recursively instantiated
        **kwargs,
    ):
        super().__init__()
        hash_embedding_dim = (
            hash_embedding.n_levels * hash_embedding.n_features_per_entry
        )
        self.hash_embedding = hash_embedding

        self.net = []
        #### Embedding layer ####
        self.net.append(self.hash_embedding)

        #### MLP layers ####
        self.net.append(nn.Linear(hash_embedding_dim, hidden_features))
        self.net.append(nn.ReLU())
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())
        if outermost_linear:
            self.net.append(nn.Linear(hidden_features, out_features))
        else:
            self.net.append(nn.Linear(hidden_features, out_features))
            self.net.append(nn.ReLU())

        self.net = nn.Sequential(*self.net)

        # Xavier initialization for the small MLP
        self.net.apply(self._init_mlp_weights)

    def _init_mlp_weights(self, module):
        """Custom initialization for the small MLP"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)

        return output, coords
