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


class HashEmbeddingMRU(HashEmbeddingUnravel):
    """Most recently (first) used (MRU) hash embedding. We maintain an MRU queue as
    "collision candidates" in case when we have to perform a hash collision. This ensures
    that coordinates collide with more recently trained coordinates, which alleviates
    forgetting because colliding with "forgotten" coordinates (those trained early on)
    leads to most interference.
    """

    def __init__(self, mru_size, **kwargs):
        """Initialize MRU queue and other book-keeping instance variables.

        Instance variables:
            used: Dictionary that contains sets of used (trained) coordinates, one for
                each resolution. Coordinates are represented by their linearly unraveled
                indices. For example, `self.used[16] = {33}` for a 2-dim grid means that
                the coordinate (1, 2) was previously trained for the 16x16 grid.
            most_recently_used: Dictionary that contains queues of most recently used
                coordinates, one for each resolution. As `used`, coordinates are
                represented by their linearly unraveled indices. Newly used coordinates
                are added to the left (front) end, and queues are cut back to a maximum
                length of `self.mru_size` from the right (tail) end.
            collision_target: Dictionary that contains hash indices for overflow
                unraveled indices, which are coordinates whose unraveled indices exceed
                the hash table size and must collide. The ith entry `index` represents
                that the coordinate with unraveled index `(i + hashtable_size)` will
                collide with `index` in the hash table. Initialized to -1. If the entry
                is not -1, the collision target had been computed before, so we directly
                use it. Else, we compute the collision target from the current MRU queue
                (see more in `spatial_hash`) and save it to `collision_target`.

        As an optimization, we only initialize these 3 instance variables for grid
        resolutions that will result in collisions (i.e. more coordinates than hash
        table size).
        """

        super().__init__(**kwargs)
        self.mru_size = mru_size

        # Set of used coordinates
        self.used = {}
        # Queue of most recently (first) used coordinates
        self.most_recently_used = {}
        # Computed collision targets
        self.collision_target = {}

        for resolution in self.strides.keys():
            if np.log2(resolution**self.coord_dim) > self.log2_hashtable_size:
                self.used[resolution] = set()
                self.most_recently_used[resolution] = deque([])
                self.collision_target[resolution] = -torch.ones(
                    int(resolution**self.coord_dim - 2**self.log2_hashtable_size),
                    dtype=torch.long,
                )

    def add_to_used(self, first_used: list, resolution: int):
        """Add a list of hash indices to `self.used`.
        Args:
            first_used: List of hash indices first used.
            resolution: Grid resolution corresponding to the hash indices.
        """
        self.used[resolution] = self.used[resolution].union(set(first_used))

    def add_to_mru(self, indices: list, resolution: int):
        """Add new `indices` to MRU. Pop tail if exceeds `self.mru_size`."""
        # Populate MRU queue
        self.most_recently_used[resolution].extendleft(indices)

        # Remove tail (right) of the MRU queue
        for _ in range(
            max(len(self.most_recently_used[resolution]) - self.mru_size, 0)
        ):
            self.most_recently_used[resolution].pop()

    def spatial_hash(self, vertex_indices: Tensor, resolution: int) -> Tensor:
        """Spatial hash function for the MRU hash scheme.
        We first unravel the d-dim coordinates into 1-dim indices.
        For indices that do not exceed the hash table size:
            1. They are directly used as hash indices for the hash embeddings
            2. For the first-used indices, we add them to `used` set and MRU queue
        For indices that exceed the hash table size:
            1. Find the collision target. For a coordinate with unraveled index `idx`,
               we look at entry `idx - hashtable_size` of `collision_target`. If the
               target is not -1, we directly use the collision target as hash index.
            2. Else, we compute spatial hash of this coordinate for a hashtale size of
               the length of the MRU queue; denote it as `mru_index`. The hash index is
               the value from the MRU queue at index `mru_index`. In other words, we
               treat the MRU queue as a hash table of hash embedding indices in case
               of a collision. We also save it as `collision_target` of this coordinate.
        """
        primes = [
            1,
            2654435761,
            805459861,
        ]

        # Unravel d-dim coordinates into indices
        stride = self.strides[resolution]
        unravel_indices = (vertex_indices * stride).sum(-1)

        overflow = unravel_indices >= (2**self.log2_hashtable_size)
        overflow_size = overflow.sum()

        # Add the non-overflow and first-use indices used set and MRU queue
        if self.training and resolution in self.used.keys():
            used = torch.tensor(list(self.used[resolution]), dtype=torch.long)
            non_overflow_indices = unravel_indices[~overflow]
            first_used = non_overflow_indices[
                torch.isin(non_overflow_indices, used, invert=True)
            ].numpy()
            self.add_to_mru(first_used, resolution)
            self.add_to_used(first_used, resolution)

        hash_indices = unravel_indices

        if overflow_size > 0:
            collision_target_entries = (
                unravel_indices[overflow] - 2**self.log2_hashtable_size
            )

            # Need to compute collision target for the first time
            need_to_compute = (
                self.collision_target[resolution][collision_target_entries] == -1
            )

            if need_to_compute.sum() > 0:
                vertex_to_hash = vertex_indices[overflow][need_to_compute]
                xor_result = torch.zeros_like(vertex_to_hash)[..., 0]
                for dim in range(self.coord_dim):
                    xor_result ^= vertex_to_hash[..., dim] * primes[dim]
                mru_indices = xor_result % len(self.most_recently_used[resolution])
                collision_targets = torch.tensor(self.most_recently_used[resolution])[
                    mru_indices
                ]

                # Use newly computed collision targets
                mask = overflow.clone()
                mask[overflow] = need_to_compute
                hash_indices[mask] = collision_targets

                # Save newly computed collision targets
                mask = collision_target_entries[need_to_compute]
                self.collision_target[resolution][mask] = collision_targets

            # Reuse collision targets
            mask = overflow.clone()
            mask[overflow] = ~need_to_compute
            hash_indices[mask] = self.collision_target[resolution][
                collision_target_entries
            ][~need_to_compute]

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
