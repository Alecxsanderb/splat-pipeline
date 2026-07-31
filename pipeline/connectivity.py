"""Image co-visibility graph: shared-point counting and connected components.

Operates on plain numpy arrays rather than COLMAP types, so it can be unit
tested against hand-built toy graphs without any model files.

Two images are "connected" when they observe at least N of the same 3D points.
The number of connected components at a chosen N is the signal that tells the
user whether their scene reconstructed as one coherent space or fragmented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class PairCounts:
    """Shared-3D-point counts for every image pair that shares at least one.

    Sub-threshold pairs are deliberately retained: they are the only way to
    later answer "which images *almost* connected", which is what makes
    boundary reporting and bridge candidates possible.
    """

    image_ids: np.ndarray  # (M,) model image ids; index i is dense id i
    pair_lo: np.ndarray  # (P,) dense id, always < pair_hi
    pair_hi: np.ndarray  # (P,) dense id
    counts: np.ndarray  # (P,) shared 3D-point count
    skipped_long_tracks: int = 0
    longest_track: int = 0

    @property
    def num_images(self) -> int:
        return int(self.image_ids.shape[0])

    @property
    def num_pairs(self) -> int:
        return int(self.counts.shape[0])

    def edges_at(self, min_common_points: int) -> tuple[np.ndarray, np.ndarray]:
        mask = self.counts >= min_common_points
        return self.pair_lo[mask], self.pair_hi[mask]


def count_shared_points(
    image_ids: np.ndarray,
    track_image_ids: np.ndarray,
    track_offsets: np.ndarray,
    *,
    max_track_length: int = 300,
) -> PairCounts:
    """Count shared 3D points for every co-visible image pair.

    `track_image_ids` is the flat CSR-style concatenation of all point tracks
    and `track_offsets` delimits them, matching `Points3D`.

    Cost is O(sum of track_length^2). Pairs are emitted with numpy per
    track-length bucket and tallied with `np.unique`, which is roughly twice as
    fast as a Counter over itertools.combinations and an order of magnitude
    lighter on memory. `np.bincount` over a dense num_images^2 key space would
    be faster still but is quadratic in image count, which stops being safe
    well before the scale this pipeline targets.
    """
    image_ids = np.asarray(image_ids)
    num_images = int(image_ids.shape[0])
    if num_images == 0:
        return PairCounts(
            image_ids=image_ids,
            pair_lo=np.zeros(0, dtype=np.int64),
            pair_hi=np.zeros(0, dtype=np.int64),
            counts=np.zeros(0, dtype=np.int64),
        )

    dense_of = {int(image_id): i for i, image_id in enumerate(image_ids.tolist())}
    track_image_ids = np.asarray(track_image_ids)
    track_offsets = np.asarray(track_offsets)

    # Group tracks by length so each bucket can emit all of its pairs at once.
    buckets: dict[int, list[list[int]]] = {}
    skipped_long = 0
    longest = 0
    for i in range(len(track_offsets) - 1):
        lo, hi = int(track_offsets[i]), int(track_offsets[i + 1])
        if hi - lo < 2:
            continue
        # A single 3D point can legitimately appear twice in one image's
        # features after a bad match; undeduped that becomes a self-pair.
        members = {dense_of[int(v)] for v in track_image_ids[lo:hi].tolist() if int(v) in dense_of}
        length = len(members)
        longest = max(longest, length)
        if length < 2:
            continue
        if length > max_track_length:
            # A repeated-texture blowup (identical doors, blank walls) can reach
            # thousands of images and contributes millions of meaningless pairs.
            skipped_long += 1
            continue
        buckets.setdefault(length, []).append(sorted(members))

    if skipped_long:
        logger.warning(
            "Skipped %d track(s) longer than %d images when building the connectivity "
            "graph (longest was %d); this usually indicates repeated texture",
            skipped_long, max_track_length, longest,
        )

    key_chunks = []
    for length, rows in buckets.items():
        block = np.array(rows, dtype=np.int64)
        rows_i, cols_i = np.triu_indices(length, k=1)
        lo = block[:, rows_i].ravel()
        hi = block[:, cols_i].ravel()
        key_chunks.append(lo * num_images + hi)

    if not key_chunks:
        return PairCounts(
            image_ids=image_ids,
            pair_lo=np.zeros(0, dtype=np.int64),
            pair_hi=np.zeros(0, dtype=np.int64),
            counts=np.zeros(0, dtype=np.int64),
            skipped_long_tracks=skipped_long,
            longest_track=longest,
        )

    keys = np.concatenate(key_chunks)
    unique_keys, counts = np.unique(keys, return_counts=True)
    return PairCounts(
        image_ids=image_ids,
        pair_lo=(unique_keys // num_images).astype(np.int64),
        pair_hi=(unique_keys % num_images).astype(np.int64),
        counts=counts.astype(np.int64),
        skipped_long_tracks=skipped_long,
        longest_track=longest,
    )


def connected_components(pairs: PairCounts, min_common_points: int) -> np.ndarray:
    """Label each image with its component id, 0 being the largest component.

    Union-find over the thresholded edge list. Isolated images (no qualifying
    edge) each form their own component, which is correct: an image sharing
    fewer than N points with everything else really is disconnected.
    """
    num_images = pairs.num_images
    parent = list(range(num_images))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    lo_edges, hi_edges = pairs.edges_at(min_common_points)
    for a, b in zip(lo_edges.tolist(), hi_edges.tolist(), strict=True):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    roots = np.array([find(i) for i in range(num_images)], dtype=np.int64)
    unique_roots, sizes = np.unique(roots, return_counts=True)
    # Relabel largest-first so component 0 is always the main reconstruction.
    order = unique_roots[np.argsort(-sizes, kind="stable")]
    relabel = {int(root): i for i, root in enumerate(order.tolist())}
    return np.array([relabel[int(r)] for r in roots], dtype=np.int64)


def component_sizes(labels: np.ndarray) -> list[int]:
    if labels.size == 0:
        return []
    _, sizes = np.unique(labels, return_counts=True)
    return sorted(sizes.tolist(), reverse=True)


def component_threshold_curve(
    pairs: PairCounts, thresholds: tuple[int, ...]
) -> list[tuple[int, int]]:
    """Component count at each threshold.

    "One component at N=5 but seven at N=30" is a very different situation from
    "one component at N=100" -- this shows whether the scene is robustly
    connected or hanging by a thread.
    """
    return [
        (threshold, int(np.unique(connected_components(pairs, threshold)).size))
        for threshold in sorted(thresholds)
    ]


@dataclass(frozen=True)
class BoundaryPair:
    """A candidate for bridging two components."""

    dense_a: int
    dense_b: int
    component_a: int
    component_b: int
    shared_points: int
    distance: float | None
    view_angle_deg: float | None
    source: str  # "shared_points" | "spatial"


def cross_component_pairs(
    pairs: PairCounts, labels: np.ndarray, min_common_points: int
) -> list[BoundaryPair]:
    """Pairs that share some 3D points but fewer than the edge threshold.

    These are the near-misses: the images that most nearly tied two components
    together. Ranked by shared count descending.
    """
    below = pairs.counts < min_common_points
    if not below.any():
        return []

    lo = pairs.pair_lo[below]
    hi = pairs.pair_hi[below]
    counts = pairs.counts[below]
    cross = labels[lo] != labels[hi]
    if not cross.any():
        return []

    lo, hi, counts = lo[cross], hi[cross], counts[cross]
    order = np.argsort(-counts, kind="stable")
    return [
        BoundaryPair(
            dense_a=int(lo[i]),
            dense_b=int(hi[i]),
            component_a=int(labels[lo[i]]),
            component_b=int(labels[hi[i]]),
            shared_points=int(counts[i]),
            distance=None,
            view_angle_deg=None,
            source="shared_points",
        )
        for i in order.tolist()
    ]


def spatial_candidates(
    centers: np.ndarray,
    directions: np.ndarray,
    labels: np.ndarray,
    component_a: int,
    component_b: int,
    *,
    neighbors_per_image: int = 3,
    max_view_angle_deg: float = 90.0,
) -> list[BoundaryPair]:
    """Nearest-camera candidates between two components sharing no 3D points.

    Valid ONLY within a single model, where every camera shares one coordinate
    frame -- never call this across separate models, whose frames are unrelated.

    Two cameras can sit 30cm apart with a wall between them, which is the most
    likely false positive in a house, so pairs whose optical axes diverge by
    more than `max_view_angle_deg` are rejected.
    """
    idx_a = np.flatnonzero(labels == component_a)
    idx_b = np.flatnonzero(labels == component_b)
    if idx_a.size == 0 or idx_b.size == 0:
        return []

    # Iterate over the smaller side so `neighbors_per_image` means what it says.
    if idx_a.size > idx_b.size:
        idx_a, idx_b = idx_b, idx_a
        component_a, component_b = component_b, component_a

    deltas = centers[idx_a][:, None, :] - centers[idx_b][None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    cos_limit = np.cos(np.radians(max_view_angle_deg))

    candidates: list[BoundaryPair] = []
    rejected = 0
    take = min(neighbors_per_image, idx_b.size)
    for row, a in enumerate(idx_a.tolist()):
        for col in np.argsort(distances[row], kind="stable")[:take].tolist():
            b = int(idx_b[col])
            cosine = float(np.dot(directions[a], directions[b]))
            if cosine < cos_limit:
                rejected += 1
                continue
            candidates.append(
                BoundaryPair(
                    dense_a=min(a, b),
                    dense_b=max(a, b),
                    component_a=int(labels[min(a, b)]),
                    component_b=int(labels[max(a, b)]),
                    shared_points=0,
                    distance=float(distances[row, col]),
                    view_angle_deg=float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
                    source="spatial",
                )
            )

    if rejected:
        logger.info(
            "Rejected %d spatial candidate(s) between components %d and %d whose cameras "
            "face away from each other", rejected, component_a, component_b,
        )
    candidates.sort(key=lambda c: c.distance if c.distance is not None else float("inf"))
    return candidates
