import time
from collections import deque
from typing import Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn

import utils


class HashEmbedding(nn.Module):
    """Multi-resolution hash function as coordinate embedding."""

    def __init__(
        self,
        coord_dim: int,
        grid_min: float,
        grid_max: float,
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
            grid_min: Minimum grid coordinate (i.e. bounding box). Usually -1.
            grid_max: Maximum grid coordinate (i.e. bounding box). Usually 1.
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
        self.grid_min = grid_min
        self.grid_max = grid_max
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

        t_adj_v = 0
        t_hash = 0
        t_interpl = 0
        # print(f"memory: {torch.cuda.memory_allocated(0) / (1024 * 1024)}")
        for i in range(self.n_levels):
            # Get adjacent vertices at this level's resolution
            t = time.time()
            resolution = np.floor(
                self.base_resolution * self.coarse_to_fine_factor**i
            )
            (
                vertex_indices,
                min_vertex_coords,
                max_vertex_coords,
            ) = utils.get_adjacent_vertices(
                x,
                grid_min=self.grid_min,
                grid_max=self.grid_max,
                vertex_resolution=resolution,
            )
            t_adj_v += time.time() - t

            # Query hash function and lookup table to get adjacent vertices' embeddings
            t = time.time()
            vertex_indices_hash = self.spatial_hash(vertex_indices, resolution)
            t_hash += time.time() - t

            vertex_embeddings = self.embeddings[i](vertex_indices_hash)

            # Interpolate adjacent vertices' embeddings to get x's embedding
            t = time.time()
            x_embedding = utils.linear_interpolate(
                x,
                min_vertex_coords,
                max_vertex_coords,
                vertex_values=vertex_embeddings,
                resolution=resolution,
            )
            x_embedding_multi_level.append(x_embedding)
            t_interpl += time.time() - t

        # print(f"t_adj_v: {t_adj_v}")
        # print(f"t_hash: {t_hash}")
        # print(f"t_interpl: {t_interpl}")

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

    @property
    def out_dim(self):
        return self.n_levels * self.n_features_per_entry


class HashEmbeddingUnravel(HashEmbedding):
    """Hash embedding where we linearly unravel the coordinate directly as hash table
    indices. We only use the spatial hash function to compute hash indices when the
    current grid resolution has more coordinates than the hash table size.

    This is consistent with the official CUDA implementation. It is almost the same as
    always using the spatial hash function, except we strictly have no collisions when
    the grid size is less than the hash table size.

    Empirically, this performs a bit better in both continual and non-continual settings.
    """

    def __init__(self, **kwargs) -> None:
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

        grid_size = resolution**self.coord_dim
        if grid_size > 2**self.log2_hashtable_size:
            xor_result = torch.zeros(
                vertex_indices.shape[:2],
                dtype=torch.long,
                device=vertex_indices.device,
            )
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

    def to(self, device: torch.device, **kwargs):
        for resolution in self.strides.keys():
            self.strides[resolution] = self.strides[resolution].to(device)
        return super().to(device, **kwargs)


class HashEmbeddingMRU(HashEmbeddingUnravel):
    """Most recently (first) used (MRU) hash embedding. We maintain an MRU queue as
    "collision candidates" in case when we have to perform a hash collision. This ensures
    that coordinates collide with more recently trained coordinates, which alleviates
    forgetting because colliding with "forgotten" coordinates (those trained early on)
    leads to most interference.
    """

    def __init__(self, mru_size, **kwargs) -> None:
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
            unravel_indices_permute: Fixed permutation for the unraveled indices, one
                for each resolution. We always permute the unraveled indices before
                querying the collision targets, which ensures that collision targets
                are not spatially biased.

        As an optimization, we only initialize these 3 instance variables for grid
        resolutions that will result in collisions (i.e. more coordinates than hash
        table size).
        """

        super().__init__(**kwargs)
        self.mru_size = int(mru_size)

        # TODO: These need to be saved as part of checkpoints!
        # Implementation plan: Additionally store buffers for each dictionary (for each
        # dict, store a keys tensor and a values tensor). Implement a `_sync_to_buffer`
        # and `_sync_from_buffer` method, which is called in `state_dict` and `load_state_dict`
        # overrides respectively.
        # Set of used coordinates
        self.used = {}
        # Queue of most recently (first) used coordinates
        self.most_recently_used = {}
        # Computed collision targets
        self.collision_target = {}
        # Fixed random permutation of unraveled indices
        self.unravel_indices_permute = {}

        for resolution in self.strides.keys():
            if resolution**self.coord_dim > 2**self.log2_hashtable_size:
                self.used[resolution] = set()
                self.most_recently_used[resolution] = deque([])
                self.collision_target[resolution] = -torch.ones(
                    int(resolution**self.coord_dim - 2**self.log2_hashtable_size),
                    dtype=torch.long,
                )
            grid_size = int(resolution**self.coord_dim)
            self.unravel_indices_permute[resolution] = torch.randperm(grid_size)

    def add_to_used(self, first_used: list, resolution: int) -> None:
        """Add a list of hash indices to `self.used`.
        Args:
            first_used: List of hash indices first used.
            resolution: Grid resolution corresponding to the hash indices.
        """
        self.used[resolution] = self.used[resolution].union(set(first_used))

    def add_to_mru(self, indices: list, resolution: int) -> None:
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
        We first unravel the d-dim coordinates into 1-dim indices. We then randomly
        permute them (permutation is fixed at initialization), which is very important
        to ensure the collision targets are not spatially biased.

        For indices that do not exceed the hash table size (non-overflow indices):
            1. They are directly used as hash indices for the hash embeddings
            2. For the first-used indices, we add them to `used` set and MRU queue
        For indices that exceed the hash table size (overflow indices):
            1. Find the collision target. For a coordinate with unraveled index `idx`,
               we look at entry `idx - hashtable_size` of `collision_target`. If the
               target is not -1, we directly use the collision target as hash index.
            2. Else, we compute spatial hash of this coordinate for a hashtale size of
               the length of the MRU queue; denote it as `mru_index`. The hash index is
               the value from the MRU queue at index `mru_index`. In other words, we
               treat the MRU queue as a hash table of hash embedding indices in case
               of a collision. We also save it as `collision_target` of this coordinate.

        Args:
            vertex_indices: (bsz, num_vertices, 2 or 3)
        Returns: hash_indices (bsz, num_vertices)
        """
        device = vertex_indices.device
        primes = [
            1,
            2654435761,
            805459861,
        ]

        # Unravel d-dim coordinates into indices
        stride = self.strides[resolution]
        unravel_indices = (vertex_indices * stride).sum(-1)

        # IMPORTANT: need to randomly permute the unraveled indices, so we don't bias
        # towards colliding with leftmost column (with small unraveled indices)!
        unravel_indices = self.unravel_indices_permute[resolution][unravel_indices]

        overflow = unravel_indices >= (2**self.log2_hashtable_size)
        overflow_size = overflow.sum()

        # Add the non-overflow and first-use indices used set and MRU queue
        if self.training and resolution in self.used.keys():
            used = torch.tensor(
                list(self.used[resolution]),
                dtype=torch.long,
                device=device,
            )
            non_overflow_indices = unravel_indices[~overflow]
            first_used = (
                non_overflow_indices[
                    torch.isin(non_overflow_indices, used, invert=True)
                ]
                .cpu()
                .numpy()
            )
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
                collision_targets = torch.tensor(
                    self.most_recently_used[resolution], device=device
                )[mru_indices]

                # Use newly computed collision targets
                mask = overflow.clone()
                mask[overflow] = need_to_compute
                hash_indices[mask] = collision_targets

                # Save newly computed collision targets ONLY IN TRAIN MODE!
                if self.training:
                    mask = collision_target_entries[need_to_compute]
                    self.collision_target[resolution][mask] = collision_targets

            # Reuse collision targets
            mask = overflow.clone()
            mask[overflow] = ~need_to_compute
            hash_indices[mask] = self.collision_target[resolution][
                collision_target_entries
            ][~need_to_compute]

        return hash_indices

    def to(self, device: torch.device, **kwargs):
        for resolution in self.collision_target.keys():
            self.collision_target[resolution] = self.collision_target[resolution].to(
                device
            )

        for resolution in self.unravel_indices_permute.keys():
            self.unravel_indices_permute[resolution] = self.unravel_indices_permute[
                resolution
            ].to(device)

        return super().to(device, **kwargs)


class HashEmbeddingMRUGrid(HashEmbeddingUnravel):
    def __init__(
        self,
        mru_size: int,
        mru_grid_resolution: int,
        permute_on_the_fly: bool,
        **kwargs,
    ) -> None:
        """A 'lossy' version of MRU hash embedding. In MRU, we need to store the computed
        hash collision targets explicitly as `self.collision_target`, which maps each
        (overflow) grid point to a collision target. This grows geometrically to the
        finest resolution. In MRUGrid, we store the computed hash collision targets
        implicitly as `self.collision_base_and_bound`, which maps each MRU voxel to a
        collision range, from which the collision target is then computed. This grows
        geometrically to the MRU grid resolution (typically much smaller than finest
        resolution in 3D). See `spatial_hash` method docstring for detailed operations.

        Note that now the MRU queue stores the entire MRU history, not just the top
        `mru_size` values. This is tractable because the MRU history is at most as large
        as the hash table size (as it only stores non-overflow indices).

        Args:
            mru_size: 'Size' of the MRU queue. This is actually used in computing bound
                for the first time, as we do no truncate the MRU.
            mru_grid_resolution: Resolution of the MRU grid.
            permute_on_the_fly: If True, compute unravel indices permutation on-the-fly
                using feistel cipher.
        """

        super().__init__(**kwargs)
        self.mru_size = int(mru_size)
        self.mru_grid_resolution = mru_grid_resolution
        self.permute_on_the_fly = permute_on_the_fly

        if permute_on_the_fly:
            # Initialize CUDA extension to compute permutation on-the-fly
            self.init_feistel_permute()
            self.init_kensler_permute()
        else:
            # Pre-compute and store the permutation of unraveled indices
            # This can be too large for 3D
            self.unravel_indices_permute = {}  # This is too large!

        # Pre-compute the strides for MRU grid (different shape than `stride`)
        self.mru_grid_stride = torch.ones([1, self.coord_dim], dtype=torch.long)
        s = 1
        for dim in range(self.coord_dim):
            self.mru_grid_stride[..., dim] = s
            s *= mru_grid_resolution

        # TODO: These need to be saved as part of checkpoints!
        # Set of used (non-collider) coordinates
        self.used = {}  # Bounded by hash table size
        # Queue of most recently (first) used coordinates
        self.most_recently_used = {}  # Bounded by hash table size
        # Map MRU voxel index to collision target range
        self.collision_base_and_bound = {}  # Bounded by `mru_grid_resolution`

        for resolution in self.strides.keys():
            if resolution**self.coord_dim > 2**self.log2_hashtable_size:
                self.used[resolution] = set()
                self.most_recently_used[resolution] = torch.empty(
                    (0,), dtype=torch.long
                )
                self.collision_base_and_bound[resolution] = -torch.ones(
                    (int(self.mru_grid_resolution**self.coord_dim), 2),
                    dtype=torch.long,
                )

            # TODO: Change to Feistel cipher implementation
            grid_size = int(resolution**self.coord_dim)
            if not permute_on_the_fly:
                self.unravel_indices_permute[resolution] = torch.randperm(grid_size)

    def add_to_used(self, first_used: list, resolution: int) -> None:
        """Add a list of hash indices to `self.used`. Exactly the same as MRU hash."""
        self.used[resolution] = self.used[resolution].union(set(first_used))

    def add_to_mru(self, indices: Tensor, resolution: int) -> None:
        """Add new `indices` to MRU. Unlike MRU hash, we do not pop the tail."""
        self.most_recently_used[resolution] = torch.cat(
            (self.most_recently_used[resolution], indices)
        )

    def vertex_to_mru_grid_indices(
        self, vertex_indices: Tensor, resolution: int
    ) -> Tensor:
        """Compute which MRU grid voxels the given vertices fall in.
        Args:
            vertex_indices: (bsz, 2 or 3)

        Returns:
            (bsz, num_vertex)
        """
        vertex_indices = (
            vertex_indices / (resolution + 1) * self.mru_grid_resolution
        ).long()
        return (vertex_indices * self.mru_grid_stride).sum(-1)

    def spatial_hash_raw(self, vertex_indices: Tensor) -> Tensor:
        """Convert 2/3-dim vertex indices to 1-dim raw hash indices without modding."""
        primes = [
            1,
            2654435761,
            805459861,
        ]
        xor_result = torch.zeros_like(vertex_indices)[..., 0]
        for dim in range(self.coord_dim):
            xor_result ^= vertex_indices[..., dim] * primes[dim]
        return xor_result

    def permute_unravel_indices(
        self, unravel_indices: Tensor, resolution: int
    ) -> Tensor:
        if self.permute_on_the_fly:
            grid_size = int(resolution**self.coord_dim)
            max_range = grid_size
            return self.kensler_permutation_gpu(unravel_indices, max_range)
        else:
            return self.unravel_indices_permute[resolution][unravel_indices]

    @torch.no_grad()
    def spatial_hash(self, vertex_indices: Tensor, resolution: int) -> Tensor:
        """Spatial hash function for the MRU grid hash scheme. This is a 'lossy' version
        of the MRU hash scheme. As the MRU hash, we first unravel the d-dim coordinates
        into 1-dim indices, permute them, and end up a set of overflow and non-overflow
        indices.

        For overflow indices, we treat them just as in the MRU hash scheme:
            1. They are directly used as hash indices for the hash embeddings
            2. For the first-used indices, we add them to `used` set and MRU queue

        For the non-overflow indices:
            1. Find the collision target:
                a. Retrieve the MRU voxel that the corresponding vertex falls into
                b. Query the `self.collision_base_and_bound` to get `base` and `bound`.
                c. If base and bound is -1, it means we need to compute its collision
                   target for the first time. Go to step 2.
                d. Else, we retrieve the collision target by hashing it into the MRU
                   queue as follows:
                   i.   Raw hash the vertex to retrieve an unbounded index `i`
                   ii.  Mod `i` by `bound` and then add `base`
                   iii. The collision target is at the i-th index of the MRU queue.
                e. Return the collision target.
            2. Compute the collision target for the first time:
                a. Get the `base` and `bound` states of the current MRU queue. Bound is
                   `self.mru_size` and base is the length of MRU queue minus bound.
                b. Compute the collision target as step 1d.
                c. Store the `base` and `bound` states *for this MRU voxel* in
                   `self.collision_base_and_bound`. Here is the 'lossy' part of this
                   scheme: all vertices within the same MRU voxel get the same base and
                   bound, even if some of them were first trained at a later base and
                   bound state!

        Args:
            vertex_indices: (bsz, num_vertices, 2 or 3)
        Returns: hash_indices (bsz, num_vertices)
        """
        device = vertex_indices.device

        # Unravel d-dim coordinates into indices
        stride = self.strides[resolution]  # (1, 1, 2 or 3)
        unravel_indices = (vertex_indices * stride).sum(-1)  # (bsz, num_vertices)

        # IMPORTANT: need to randomly permute the unraveled indices, so we don't bias
        # towards colliding with leftmost column (with small unraveled indices)!
        t_permute = time.time()
        unravel_indices = self.permute_unravel_indices(unravel_indices, resolution)
        t_permute -= time.time()

        overflow = unravel_indices >= (
            2**self.log2_hashtable_size
        )  # (bsz, num_vertices)
        overflow_size = overflow.sum()

        # Add the non-overflow and first-use indices to used set and MRU queue
        t_add = time.time()
        if self.training and resolution in self.used.keys():
            used = torch.tensor(
                list(self.used[resolution]),
                dtype=torch.long,
                device=device,
            )
            non_overflow_indices = unravel_indices[~overflow]
            first_used = non_overflow_indices[
                torch.isin(non_overflow_indices, used, invert=True)
            ]
            self.add_to_mru(first_used, resolution)
            self.add_to_used(first_used.cpu().numpy(), resolution)
        t_add -= time.time()

        hash_indices = unravel_indices  # (bsz, num_vertices)

        if overflow_size > 0:
            t_v2mru = time.time()
            # TODO: Is it faster to store `overflow_vertex_indices` first?
            # Get which voxel in the MRU grid each vertex falls into
            mru_grid_indices = self.vertex_to_mru_grid_indices(
                vertex_indices[overflow], resolution
            )  # (num_overflow, )
            t_v2mru -= time.time()

            # Need to compute collision target for the first time
            t_compute = time.time()
            need_to_compute = (
                self.collision_base_and_bound[resolution][mru_grid_indices, 0] == -1
            )  # (num_to_compute, )

            if need_to_compute.sum() > 0:
                bound = min(self.mru_size, len(self.most_recently_used[resolution]))
                base = len(self.most_recently_used[resolution]) - bound
                mru_indices = (
                    self.spatial_hash_raw(vertex_indices[overflow][need_to_compute])
                    % bound
                    + base
                )

                # In case we need to evaluate before any training (and so MRU is empty)
                if self.most_recently_used[resolution].numel() == 0:
                    assert not self.training
                    collision_targets = torch.randint(
                        2**self.log2_hashtable_size, mru_indices.shape, device=device
                    )
                else:
                    collision_targets = self.most_recently_used[resolution][mru_indices]

                # Use newly computed collision targets
                mask = overflow.clone()  # Boolean mask
                mask[overflow] = need_to_compute
                hash_indices[mask] = collision_targets

                # Save newly computed collision targets ONLY IN TRAIN MODE!
                if self.training:
                    mask = mru_grid_indices[need_to_compute]  # Index mask
                    base_and_bound = torch.tensor(
                        [[base, bound]], dtype=torch.long, device=device
                    ).repeat(need_to_compute.sum(), 1)
                    self.collision_base_and_bound[resolution][mask] = base_and_bound
            t_compute -= time.time()

            # Reuse collision targets
            # TODO: Is it faster to do mru_grid_indices[~need_to_compute] instead?
            t_reuse = time.time()
            base_and_bound = self.collision_base_and_bound[resolution][
                mru_grid_indices
            ][~need_to_compute]
            base = base_and_bound[:, 0]
            bound = base_and_bound[:, 1]
            # Compute collision targets from base and bound on-the-fly
            mru_indices = (
                self.spatial_hash_raw(vertex_indices[overflow][~need_to_compute])
                % bound
                + base
            )
            collision_targets = self.most_recently_used[resolution][mru_indices]

            mask = overflow.clone()  # Boolean mask
            mask[overflow] = ~need_to_compute
            hash_indices[mask] = collision_targets
            t_reuse -= time.time()

        #     print(f"t_v2mru: {t_v2mru}")
        #     print(f"t_compute: {t_compute}")
        #     print(f"t_reuse: {t_reuse}")

        # print(f"t_permute: {t_permute}")
        # print(f"t_add: {t_add}")

        return hash_indices

    def init_feistel_permute(self):
        """Compile CUDA implementation of Feistel cipher for on-the-fly permutation."""
        import cupy as cp

        feistel_kernel = cp.RawKernel(
            r"""
            __device__ unsigned int round_function(unsigned int block, unsigned int key) {
                unsigned int hash = (block ^ key) + block * key;
                return hash;
            }

            extern "C" __global__
            void feistel_permutation_kernel(const unsigned int* input, unsigned int* output,
                                            const unsigned int key1, const unsigned int key2,
                                            int rounds, const int size, const unsigned int range) {
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                if (idx >= size) {
                    return;
                }

                unsigned int value = input[idx];
                unsigned int mask = (1 << 16) - 1;
                unsigned int left = value >> 16;
                unsigned int right = value & mask;
                
                unsigned int temp_key1 = key1;
                unsigned int temp_key2 = key2;

                for (int i = 0; i < rounds; ++i) {
                    // Swap left and right mask
                    unsigned int temp = left;
                    left = right;
                    right = temp ^ round_function(right, temp_key1) & mask;
                    
                    // Swap key 1 and key 2
                    unsigned int temp_key = temp_key1;
                    temp_key1 = temp_key2;
                    temp_key2 = temp_key;

                    if (i == rounds - 1) {
                        // At last round
                        unsigned int temp_result = (left << 16) | right;
                        if (temp_result >= range) {
                            rounds = rounds + 1;
                        }
                    }
                }

                output[idx] = (left << 16) | right;

            }
            """,
            "feistel_permutation_kernel",
        )

        def feistel_permutation_gpu(
            tensor: Tensor,
            max_range: int,
            key1: int = 12345,
            key2: int = 67890,
            rounds: int = 2,
        ) -> Tensor:
            input_gpu = cp.asarray(tensor, dtype=cp.uint32)
            # input_gpu = tensor.astype(torch.uint32)
            output_gpu = cp.empty_like(input_gpu)
            # output_gpu = torch.empty_like(input_gpu)
            size = input_gpu.size

            threads_per_block = 256
            blocks = (size + threads_per_block - 1) // threads_per_block

            feistel_kernel(
                (blocks,),
                (threads_per_block,),
                (
                    input_gpu,
                    output_gpu,
                    key1,
                    key2,
                    rounds,
                    size,
                    max_range,
                ),
            )

            return torch.as_tensor(
                output_gpu.astype(cp.int64), device=tensor.device, dtype=torch.long
            )

        self.feistel_permutation_gpu = feistel_permutation_gpu

    def init_kensler_permute(self):
        import cupy as cp

        kensler_kernel = cp.RawKernel(
            r"""
            extern "C" __global__
            void kensler_permutation_kernel(const unsigned int* input, unsigned int* output,
                                            const unsigned int key, const int size, 
                                            const unsigned int range) {
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                if (idx >= size) {
                    return;
                }

                unsigned int w = range - 1;
                w |= w >> 1;
                w |= w >> 2;
                w |= w >> 4;
                w |= w >> 8;
                w |= w >> 16;

                unsigned int i = input[idx];

                do {
                    i ^= key;
                    i *= 0xe170893d;
                    i ^= key >> 16;
                    i ^= (i & w) >> 4;
                    i ^= key >> 8;
                    i *= 0x0929eb3f;
                    i ^= key >> 23;
                    i ^= (i & w) >> 1;
                    i *= 1 | key >> 27;
                    i *= 0x6935fa69;
                    i ^= (i & w) >> 11;
                    i *= 0x74dcb303;
                    i ^= (i & w) >> 2;
                    i *= 0x9e501cc3;
                    i ^= (i & w) >> 2;
                    i *= 0xc860a3df;
                    i &= w;
                    i ^= i >> 5;
                } while (i >= range);
                
                output[idx] = (i + key) % range;
            }
            """,
            "kensler_permutation_kernel",
        )

        def kensler_permutation_gpu(tensor, max_range, key=2654435761):
            if tensor.numel() == 0:
                return tensor

            input_gpu = cp.asarray(tensor, dtype=cp.uint32)
            output_gpu = cp.empty_like(input_gpu)
            size = input_gpu.size

            threads_per_block = 256
            blocks = (size + threads_per_block - 1) // threads_per_block

            kensler_kernel(
                (blocks,),
                (threads_per_block,),
                (input_gpu, output_gpu, key, size, max_range),
            )

            return torch.as_tensor(
                output_gpu.astype(cp.int64), device=tensor.device, dtype=torch.long
            )

        self.kensler_permutation_gpu = kensler_permutation_gpu

    def to(self, device: torch.device, **kwargs):
        self.mru_grid_stride = self.mru_grid_stride.to(device)

        for resolution in self.collision_base_and_bound.keys():
            self.most_recently_used[resolution] = self.most_recently_used[
                resolution
            ].to(device)
            self.collision_base_and_bound[resolution] = self.collision_base_and_bound[
                resolution
            ].to(device)

        if not self.permute_on_the_fly:
            for resolution in self.unravel_indices_permute.keys():
                self.unravel_indices_permute[resolution] = self.unravel_indices_permute[
                    resolution
                ].to(device)

        return super().to(device, **kwargs)


class HashNet(nn.Module):
    """Network that uses a multi-resolution hash function as coordinate embedding."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool,
        use_tcnn: bool,
        hash_embedding: HashEmbedding,  # Recursively instantiated
        **kwargs,
    ) -> None:
        super().__init__()
        hash_embedding_dim = hash_embedding.out_dim
        self.hash_embedding = hash_embedding
        self._out_dim = out_features

        self.net = []
        #### Embedding layer ####
        self.net.append(self.hash_embedding)

        #### MLP layers ####
        if use_tcnn:
            import tinycudann as tcnn

            network_config = {
                "otype": "FullyFusedMLP",  # Component type.
                "activation": "ReLU",  # Activation of hidden layers.
                "output_activation": "None"
                if outermost_linear
                else "ReLU",  # Activation of the output layer.
                "n_neurons": hidden_features,  # Neurons in each hidden layer.
                # May only be 16, 32, 64, or 128.
                "n_hidden_layers": hidden_layers,  # Number of hidden layers.
            }

            self.net.append(
                tcnn.Network(
                    n_input_dims=hash_embedding_dim,
                    n_output_dims=out_features,
                    network_config=network_config,
                    seed=0,  # TODO: Read from config?
                )
            )

        else:
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

    def _init_mlp_weights(self, module: nn.Module) -> None:
        """Custom initialization for the small MLP"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, coords: Tensor, max_batch: int = 1024**2) -> Tensor:
        """Forward pass to get RGB values given coordinates. If batch size exceeds
        `max_batch`, do multiple forward passes.
        """
        bsz = coords.shape[0]
        if coords.shape[0] > max_batch:
            out = torch.empty(
                (bsz, self.out_dim), dtype=torch.float32, device=coords.device
            )
            i = 0
            while i < bsz:
                out_batch = self.net(coords[i : min(i + max_batch, bsz)])
                out[i : min(i + max_batch, bsz)] = out_batch
                i += max_batch
            return out
        else:
            return self.net(coords)

    def to(self, device: torch.device, **kwargs):
        self.hash_embedding.to(device, **kwargs)
        return super().to(device, **kwargs)

    @property
    def out_dim(self):
        return self._out_dim


class HashEmbeddingUnravelBlock(HashEmbedding):
    """Unravel hash embedding scheme for block HashNet. Global coordinates are shifted
    to the local coordinate system, and grid size is adjusted when unraveling.
    """

    def __init__(self, local_shift: Tensor, local_scale: Tensor, **kwargs) -> None:
        """
        Args:
            local_shift: (1, D) Shift to apply on global coordinate to get local
                coordinate in [-1, 1].
            local_scale: (1, D) Scale to apply on global coordinate to get local
                coordinate in [-1, 1].

        Note that `resolution` refers to the *local* block's grid resolution! For
        example, if we have 2x2 block HashNets, then a 8x8 local grid resolution
        corresponds to 16x16 global resolution.
        """
        self.local_shift = local_shift
        self.local_scale = local_scale
        super().__init__(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Shift `x` into local coordinate first before querying."""
        x = (x + self.local_shift) * self.local_scale
        return super().forward(x)

    def to(self, device: torch.device, **kwargs):
        self.local_shift = self.local_shift.to(device)
        self.local_scale = self.local_scale.to(device)
        return super().to(device, **kwargs)


class BlockHashNet(nn.Module):
    """Block version of HashNet where the total space is split into regions, and one
    HashNet is responsible for one region.
    """

    def __init__(
        self,
        bound_min: float,
        bound_max: float,
        num_blocks_per_side: int,
        out_features: int,
        hash_embedding: DictConfig,  # NOT recursively instantiated
        **kwargs,
    ) -> None:
        """Initialize by setting up a list of HashNets.
        Args:
            bound_min: Minimum bound of the total space (usually -1).
            bound_max: Maximum bound of the total space (usually 1).
            num_blocks_per_side: Number of block HashNets per side. If in 2D, then
                a total of `num_blocks_per_side ** 2` HashNets will be created.
            hash_embedding: DictConfig for hash embedding of *one* block HashNet. This
                will be instantiated once per block HashNet.
        """
        super().__init__()

        self.bound_min = bound_min
        self.bound_max = bound_max
        self.num_blocks_per_side = num_blocks_per_side
        self.block_side_length = (bound_max - bound_min) / num_blocks_per_side
        self.out_features = out_features
        self.coord_dim = int(hash_embedding["coord_dim"])

        # Initialize block HashNets
        self.num_blocks_total = num_blocks_per_side**self.coord_dim
        self.block_hash_nets = nn.ModuleList()

        assert self.coord_dim == 2
        for i in range(self.num_blocks_per_side):
            for j in range(self.num_blocks_per_side):
                local_shift = torch.tensor(
                    [[-1 + self.block_side_length * i, -1 + self.block_side_length * j]]
                )
                local_scale = torch.tensor(
                    [[self.block_side_length, self.block_side_length]]
                )

                embedding = instantiate(
                    hash_embedding, local_shift=local_shift, local_scale=local_scale
                )
                hash_net = HashNet(
                    **kwargs, out_features=out_features, hash_embedding=embedding
                )
                self.block_hash_nets.append(hash_net)

        # Pre-compute strides for unraveling region indices
        self.stride = torch.ones([1, self.coord_dim], dtype=torch.long)
        s = 1
        for dim in range(self.coord_dim):
            self.stride[..., dim] = s
            s *= num_blocks_per_side

    def forward(self, coords: Tensor) -> Tensor:
        """Forward pass groups the batch of `coords` by the blocks they fall in, and
        queries the corresponding HashNet for that block.
        """
        # Assign region index to each coordinate
        block_indices = self.coords_to_block_indices(coords)

        # Query HashNet block by block
        output = torch.zeros((coords.shape[0], self.out_features), device=coords.device)

        for i in range(self.num_blocks_total):
            mask = block_indices == i
            if mask.sum() > 0:
                coords_in_block = coords[mask]
                output_block = self.block_hash_nets[i](coords_in_block)
                output[mask] = output_block

        return output

    def coords_to_block_indices(self, coords: Tensor) -> Tensor:
        """Convert a batch of coordinates to a batch of region indices.
        Args:
            coords: (bsz, self.coord_dim) Batch of coordinates.

        Returns: (bsz, ) Batch of region indices in [0, num_blocks_total - 1]
        """
        blocks = ((coords - self.bound_min) / self.block_side_length).int()

        # Need to clamp for the edge case when coord is exactly 1 (bound_max)
        blocks = torch.clamp(blocks, max=self.num_blocks_per_side - 1)

        block_indices = (blocks * self.stride).sum(-1)
        return block_indices

    def to(self, device: torch.device, **kwargs):
        for hash_net in self.block_hash_nets:
            hash_net.to(device)
        self.stride = self.stride.to(device)
        return super().to(device, **kwargs)


def main():
    import math

    import cupy as cp
    import torch

    feistel_kernel = cp.RawKernel(
        r"""
        __device__ unsigned int round_function(unsigned int block, unsigned int key) {
            unsigned int hash = (block ^ key) + block * key;
            return hash;
        }

        extern "C" __global__
        void feistel_permutation_kernel(const unsigned int* input, unsigned int* output,
                                        const unsigned int key1, const unsigned int key2,
                                        const int rounds, const int size, const int range_power) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= size) {
                return;
            }

            unsigned int value = input[idx];
            unsigned int half_range_power = range_power / 2;
            unsigned int right_mask = (1 << half_range_power) - 1;
            unsigned int left_mask = (1 << (range_power - half_range_power)) - 1;
            
            unsigned int right = value & right_mask;
            unsigned int left = (value >> half_range_power);
            
            unsigned int temp_key1 = key1;
            unsigned int temp_key2 = key2;

            for (int i = 0; i < rounds; ++i) {
                // Swap left and right mask
                unsigned int temp_mask = left_mask;
                left_mask = right_mask;
                right_mask = temp_mask;

                // Swap left with right
                unsigned int temp = left;
                left = right & left_mask;

                // Compute right with round function and old left
                right = temp ^ round_function(right, temp_key1);
                right = right & right_mask;
                
                // Swap key 1 and key 2
                unsigned int temp_key = temp_key1;
                temp_key1 = temp_key2;
                temp_key2 = temp_key;
            }

            // NOTE: This only works if rounds is even or range_power is even
            output[idx] = (left << half_range_power) | right;
        }
        """,
        "feistel_permutation_kernel",
    )

    log2_range = 4

    def feistel_permutation_gpu(tensor, key1, key2, rounds=2):
        input_gpu = cp.asarray(tensor, dtype=cp.uint32)
        # input_gpu = tensor
        output_gpu = cp.empty_like(input_gpu)
        # output_gpu = torch.empty_like(input_gpu)
        size = input_gpu.size
        # size = input_gpu.numel()

        threads_per_block = 256
        blocks = (size + threads_per_block - 1) // threads_per_block

        feistel_kernel(
            (blocks,),
            (threads_per_block,),
            (input_gpu, output_gpu, key1, key2, rounds, size, log2_range),
        )

        return torch.tensor(output_gpu.get().astype(np.int32), dtype=torch.int32).cuda()

    # Example usage
    key1 = 12345
    key2 = 67890
    rounds = 2
    tensor = torch.randint(0, 2**3, (10,), dtype=torch.int32).cuda()

    print(f"Input tensor: {tensor}")

    permuted_tensor = feistel_permutation_gpu(tensor, key1, key2, rounds)
    print("Permuted tensor:\n", permuted_tensor)

    # # To reverse the permutation, just call the function again
    # original_tensor = feistel_permutation_gpu(
    #     permuted_tensor.to(dtype=torch.int32), key1, key2, rounds
    # )
    # print("Original tensor:\n", original_tensor)


def main2():
    import hashlib

    def deterministic_permutation_map(x):
        # Convert x to bytes and hash it using SHA-256
        hashed = hashlib.sha256(str(x).encode()).digest()

        # Convert the hashed bytes to an integer using the first 4 bytes
        # of the hash (32 bits)
        hashed_int = int.from_bytes(hashed[:4], byteorder="big")

        # Map the hashed integer to the desired range using modular arithmetic
        # Note that the range [0, 123456] has 123457 elements
        return hashed_int % 123457

    # Test the function
    x = torch.arange(123457)
    import ipdb

    ipdb.set_trace()

    y = deterministic_permutation_map(x)
    print(y)  # tensor([27545, 43510, 70114,  ..., 85238, 49560, 43127])


def main3():
    import cupy as cp
    import torch

    kensler_kernel = cp.RawKernel(
        r"""
        extern "C" __global__
        void kensler_permutation_kernel(const unsigned int* input, unsigned int* output,
                                        const unsigned int key, const int size, 
                                        const unsigned int range) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= size) {
                return;
            }

            unsigned int w = range - 1;
            w |= w >> 1;
            w |= w >> 2;
            w |= w >> 4;
            w |= w >> 8;
            w |= w >> 16;

            unsigned int i = input[idx];

            do {
                i ^= key;
                i *= 0xe170893d;
                i ^= key >> 16;
                i ^= (i & w) >> 4;
                i ^= key >> 8;
                i *= 0x0929eb3f;
                i ^= key >> 23;
                i ^= (i & w) >> 1;
                i *= 1 | key >> 27;
                i *= 0x6935fa69;
                i ^= (i & w) >> 11;
                i *= 0x74dcb303;
                i ^= (i & w) >> 2;
                i *= 0x9e501cc3;
                i ^= (i & w) >> 2;
                i *= 0xc860a3df;
                i &= w;
                i ^= i >> 5;
            } while (i >= range);
            
            // return (i + key) % l;
            output[idx] = (i + key) % range;
        }
        """,
        "kensler_permutation_kernel",
    )

    max_range = 256

    def kensler_permutation_gpu(tensor, key1):
        input_gpu = cp.asarray(tensor, dtype=cp.uint32)
        output_gpu = cp.empty_like(input_gpu)
        size = input_gpu.size

        threads_per_block = 256
        blocks = (size + threads_per_block - 1) // threads_per_block

        kensler_kernel(
            (blocks,),
            (threads_per_block,),
            (input_gpu, output_gpu, key1, size, max_range),
        )

        return torch.as_tensor(
            output_gpu.astype(cp.int32), device=tensor.device, dtype=torch.long
        )

    # Example usage
    key = 12345
    tensor = torch.randint(0, 2**3, (10,), dtype=torch.int32).cuda()

    print(f"Input tensor: {tensor}")

    permuted_tensor = kensler_permutation_gpu(tensor, key)
    print("Permuted tensor:\n", permuted_tensor)

    # # To reverse the permutation, just call the function again
    # original_tensor = feistel_permutation_gpu(
    #     permuted_tensor.to(dtype=torch.int32), key1, key2, rounds
    # )
    # print("Original tensor:\n", original_tensor)


if __name__ == "__main__":
    main3()
