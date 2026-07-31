"""chunk: carve a reconstruction into per-room sub-models that fit in VRAM.

Each named box becomes a self-contained COLMAP model holding the images that
see that room and the 3D points inside it, plus a manifest recording the box
and the counts. `merge` later crops each trained chunk back to its manifest
box so overlapping chunks do not leave doubled geometry at the walls.

Exit codes:
    0  every requested chunk was written
    1  at least one chunk could not be written
    3  nothing to chunk (no model, or no box definitions)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pipeline.boxes import BoundingBox, BoxError, bounds_of, load_boxes, save_boxes
from pipeline.colmap_model import (
    ColmapModelError,
    Model,
    find_models,
    read_model,
    subset_model,
    write_model,
)
from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOTHING_TO_DO = 3

MANIFEST_NAME = "manifest.json"
INDEX_NAME = "chunks.json"


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("chunk", help="Split reconstruction into per-room chunks")
    # Every argument must stay optional; `splat chunk` is exercised bare by the
    # CLI tests.
    parser.add_argument("--boxes", type=Path, default=None,
                        help="YAML file of named bounding boxes")
    parser.add_argument("--sparse-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", type=int, default=None,
                        help="Use this sparse submodel instead of the largest")
    parser.add_argument("--min-points-in-box", type=int, default=None,
                        help="Points inside the box an outside image must observe to join")
    parser.add_argument("--suggest", action="store_true",
                        help="Propose boxes by clustering camera positions and exit")
    parser.add_argument("--suggest-method", choices=["dbscan", "kmeans"], default=None)
    parser.add_argument("--clusters", type=int, default=None,
                        help="k, when suggesting with kmeans")
    parser.add_argument("--eps", type=float, default=None,
                        help="DBSCAN neighbourhood radius, in model units")
    return parser


@dataclass(frozen=True)
class ChunkOptions:
    boxes_path: Path
    sparse_dir: Path
    output_dir: Path
    model_index: int | None
    min_points_in_box: int
    point_margin: float
    min_track_length: int
    max_images_per_chunk: int
    target_gaussians_min: int
    target_gaussians_max: int
    suggest: bool
    suggest_method: str
    suggest_eps: float
    suggest_min_samples: int
    suggest_clusters: int
    suggest_percentile: float
    suggest_padding: float


def _resolve_options(args: argparse.Namespace, config: PipelineConfig) -> ChunkOptions:
    cfg = config.chunk

    def pick(name: str, fallback):
        value = getattr(args, name, None)
        return fallback if value is None else value

    return ChunkOptions(
        boxes_path=pick("boxes", config.paths.output_dir / "chunks" / "boxes.yaml"),
        sparse_dir=pick("sparse_dir", config.paths.output_dir / "sparse"),
        output_dir=pick("output_dir", config.paths.output_dir / "chunks"),
        model_index=getattr(args, "model", None),
        min_points_in_box=pick("min_points_in_box", cfg.min_points_in_box),
        point_margin=cfg.point_margin,
        min_track_length=cfg.min_track_length,
        max_images_per_chunk=cfg.max_images_per_chunk,
        target_gaussians_min=cfg.target_gaussians_min,
        target_gaussians_max=cfg.target_gaussians_max,
        suggest=bool(getattr(args, "suggest", False)),
        suggest_method=pick("suggest_method", cfg.suggest_method),
        suggest_eps=pick("eps", cfg.suggest_eps),
        suggest_min_samples=cfg.suggest_min_samples,
        suggest_clusters=pick("clusters", cfg.suggest_clusters),
        suggest_percentile=cfg.suggest_percentile,
        suggest_padding=cfg.suggest_padding,
    )


@dataclass
class Selection:
    """Which images and points belong to one box, and why."""

    box: BoundingBox
    image_ids: list[int]
    point_ids: list[int]
    cameras_inside: int
    through_observers: int
    observations_in_box: dict[int, int]


def select_for_box(
    model: Model,
    box: BoundingBox,
    *,
    min_points_in_box: int,
    point_margin: float = 0.0,
) -> Selection:
    """Pick the images and 3D points that make up one room's chunk.

    Two kinds of image qualify:

    * cameras physically standing inside the box, and
    * cameras anywhere else that observe at least `min_points_in_box` of the
      box's points -- which is what captures a view from the hallway looking
      in through a doorway. Without those, a room trains with no coverage of
      the wall containing its own door.
    """
    points = model.points3d
    selection_box = box.expanded(point_margin) if point_margin else box

    if len(points):
        inside = selection_box.contains(points.xyz)
    else:
        inside = np.zeros(0, dtype=bool)
    point_ids = [int(pid) for pid in points.ids[inside].tolist()]
    point_id_set = set(point_ids)

    # How many in-box points each image observes, straight off the CSR tracks.
    observations: dict[int, int] = {}
    inside_indices = np.flatnonzero(inside)
    for i in inside_indices.tolist():
        lo, hi = int(points.track_offsets[i]), int(points.track_offsets[i + 1])
        for image_id in np.unique(points.track_image_ids[lo:hi]).tolist():
            observations[int(image_id)] = observations.get(int(image_id), 0) + 1

    image_ids, centers = model.camera_centers()
    cameras_inside: set[int] = set()
    if len(image_ids):
        mask = box.contains(centers)
        cameras_inside = {image_ids[i] for i in np.flatnonzero(mask).tolist()}

    through = {
        image_id
        for image_id, count in observations.items()
        if count >= min_points_in_box and image_id not in cameras_inside
    }

    selected = sorted(cameras_inside | through)
    return Selection(
        box=box,
        image_ids=selected,
        point_ids=sorted(point_id_set),
        cameras_inside=len(cameras_inside),
        through_observers=len(through),
        observations_in_box=observations,
    )


def _cluster(centers: np.ndarray, options: ChunkOptions) -> np.ndarray:
    """Cluster camera centres into candidate rooms. Returns a label per camera."""
    try:
        from sklearn.cluster import DBSCAN, KMeans
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ColmapModelError(
            f"--suggest needs scikit-learn ({exc}); install it with `pip install scikit-learn`"
        ) from exc

    if options.suggest_method == "kmeans":
        k = min(options.suggest_clusters, centers.shape[0])
        logger.info("Clustering %d camera positions with k-means (k=%d)", len(centers), k)
        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(centers)

    logger.info(
        "Clustering %d camera positions with DBSCAN (eps=%.3f, min_samples=%d)",
        len(centers), options.suggest_eps, options.suggest_min_samples,
    )
    return DBSCAN(eps=options.suggest_eps, min_samples=options.suggest_min_samples).fit_predict(
        centers
    )


def _assign_points_to_clusters(points, image_ids: list[int], labels: np.ndarray) -> np.ndarray:
    """Give each 3D point to the cluster whose cameras observe it most.

    Sizing a cluster's box from *every* point its cameras can see would let one
    camera angled through a doorway drag the whole neighbouring room into the
    box -- the suggested rooms would then overlap almost completely and be
    useless as chunks. A majority vote keeps a point with the room it really
    belongs to: the doorway camera is one vote against the several cameras
    standing in the room itself.
    """
    label_of_image = {
        image_id: int(label)
        for image_id, label in zip(image_ids, labels.tolist(), strict=True)
    }
    owner = np.full(len(points), -1, dtype=np.int64)
    for i in range(len(points)):
        lo, hi = int(points.track_offsets[i]), int(points.track_offsets[i + 1])
        votes: dict[int, int] = {}
        for image_id in points.track_image_ids[lo:hi].tolist():
            label = label_of_image.get(int(image_id), -1)
            if label >= 0:  # cameras DBSCAN called noise do not vote
                votes[label] = votes.get(label, 0) + 1
        if votes:
            # Ties break toward the lower label, so the result is deterministic.
            owner[i] = max(sorted(votes), key=lambda k: votes[k])
    return owner


def suggest_boxes(model: Model, options: ChunkOptions) -> list[BoundingBox]:
    """Propose one box per camera cluster, sized to the geometry it observes."""
    image_ids, centers = model.camera_centers()
    if centers.shape[0] < 2:
        raise BoxError(f"need at least 2 registered cameras to suggest boxes, got {len(centers)}")

    labels = _cluster(centers, options)
    unique = sorted({int(v) for v in labels.tolist() if v >= 0})
    noise = int(np.count_nonzero(labels < 0))
    if noise:
        logger.warning(
            "%d camera(s) were not assigned to any cluster and are not covered by a box; "
            "lower --eps or raise chunk.suggest_min_samples if that seems wrong", noise,
        )
    if not unique:
        raise BoxError(
            "clustering produced no clusters at all -- try a larger --eps "
            f"(currently {options.suggest_eps}) or --suggest-method kmeans"
        )

    points = model.points3d
    owner = _assign_points_to_clusters(points, image_ids, labels)

    boxes: list[BoundingBox] = []
    for label in unique:
        # A box around camera positions alone would sit inside the room and
        # exclude its walls, so grow it to cover the geometry those cameras
        # see, trimmed by percentile against stray triangulations.
        observed = np.flatnonzero(owner == label)
        cluster_centers = centers[np.flatnonzero(labels == label)]
        if observed.size:
            lo_pt, hi_pt = bounds_of(points.xyz[observed], options.suggest_percentile)
            lo = np.minimum(cluster_centers.min(axis=0), lo_pt)
            hi = np.maximum(cluster_centers.max(axis=0), hi_pt)
        else:
            lo, hi = cluster_centers.min(axis=0), cluster_centers.max(axis=0)

        pad = options.suggest_padding
        boxes.append(
            BoundingBox(
                name=f"room_{label:02d}",
                min=tuple((lo - pad).tolist()),
                max=tuple((hi + pad).tolist()),
            )
        )
    return boxes


def _run_suggest(model: Model, options: ChunkOptions) -> int:
    boxes = suggest_boxes(model, options)

    logger.info("Proposed %d box(es):", len(boxes))
    rows = []
    for box in boxes:
        inside = int(np.count_nonzero(box.contains(model.points3d.xyz)))
        extent = box.extent
        rows.append((box, inside))
        logger.info(
            "  %-12s %7d points  extent %.2f x %.2f x %.2f",
            box.name, inside, extent[0], extent[1], extent[2],
        )

    # Sparse points are a rough proxy for trained Gaussian count, so flag boxes
    # that look likely to blow the VRAM budget (or to be barely worth training).
    for box, inside in rows:
        if inside > options.target_gaussians_max:
            logger.warning(
                "  box %r holds %d sparse points, above the %d target -- training may exceed "
                "VRAM; consider splitting it", box.name, inside, options.target_gaussians_max,
            )
        elif inside < 100:
            logger.warning("  box %r holds only %d sparse points", box.name, inside)

    save_boxes(
        options.boxes_path,
        boxes,
        header=(
            "Suggested by `splat chunk --suggest`. Rename the rooms and adjust the bounds, "
            "then re-run `splat chunk --boxes " + options.boxes_path.as_posix() + "`."
        ),
    )
    logger.info("Wrote %s", options.boxes_path)
    logger.info("Review and rename the boxes, then run: splat chunk --boxes %s",
                options.boxes_path)
    return EXIT_OK


def _pick_model(options: ChunkOptions) -> tuple[Path, Model] | None:
    try:
        model_paths = find_models(options.sparse_dir)
    except ColmapModelError as exc:
        logger.error("%s", exc)
        return None

    # points2d are required: a subset model has to rewrite each image's feature
    # table to drop references to points that left the chunk.
    try:
        models = [read_model(path, read_points2d=True) for path in model_paths]
    except ColmapModelError as exc:
        logger.error("Could not read the sparse model: %s", exc)
        return None

    if options.model_index is not None:
        for path, model in zip(model_paths, models, strict=True):
            if path.name == str(options.model_index):
                return path, model
        logger.error("No submodel named %s in %s", options.model_index, options.sparse_dir)
        return None

    if len(models) > 1:
        logger.warning(
            "%d models found in %s; chunking the largest. Chunks from different models "
            "cannot be merged, because their coordinate frames are unrelated.",
            len(models), options.sparse_dir,
        )
    return max(
        zip(model_paths, models, strict=True),
        key=lambda pair: pair[1].num_registered_images,
    )


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    options = _resolve_options(args, config)

    picked = _pick_model(options)
    if picked is None:
        return EXIT_NOTHING_TO_DO
    model_path, model = picked
    logger.info(
        "Chunking %s (%d images, %d points)",
        model_path, model.num_registered_images, model.num_points3d,
    )

    if options.suggest:
        try:
            return _run_suggest(model, options)
        except BoxError as exc:
            logger.error("Could not suggest boxes: %s", exc)
            return EXIT_FAILED

    try:
        boxes = load_boxes(options.boxes_path)
    except BoxError as exc:
        logger.error("%s", exc)
        logger.error("Generate a starting point with: splat chunk --suggest")
        return EXIT_NOTHING_TO_DO

    logger.info("Loaded %d box(es) from %s", len(boxes), options.boxes_path)
    options.output_dir.mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, object]] = []
    failed = 0

    for box in boxes:
        selection = select_for_box(
            model, box,
            min_points_in_box=options.min_points_in_box,
            point_margin=options.point_margin,
        )
        if not selection.image_ids:
            logger.error(
                "  %-12s no images select into this box -- check its bounds are in the "
                "model's coordinate frame", box.name,
            )
            failed += 1
            continue
        if not selection.point_ids:
            logger.error("  %-12s contains no 3D points", box.name)
            failed += 1
            continue

        try:
            chunk_dir = options.output_dir / box.name
            submodel, stats = subset_model(
                model,
                set(selection.image_ids),
                set(selection.point_ids),
                min_track_length=options.min_track_length,
                path=chunk_dir / "sparse" / "0",
            )
            write_model(chunk_dir / "sparse" / "0", submodel)
        except ColmapModelError as exc:
            logger.error("  %-12s could not be written: %s", box.name, exc)
            failed += 1
            continue

        manifest = {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "name": box.name,
            "box": box.to_dict(),
            "source_model": model_path.as_posix(),
            "num_images": stats.images,
            "num_cameras": stats.cameras,
            "num_points": stats.points,
            "num_observations": stats.observations,
            "cameras_inside_box": selection.cameras_inside,
            "through_doorway_observers": selection.through_observers,
            "min_points_in_box": options.min_points_in_box,
            "points_dropped_short_track": stats.points_dropped_short_track,
            "images": sorted(img.name for img in submodel.images.values()),
        }
        (chunk_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        manifests.append({k: v for k, v in manifest.items() if k != "images"})

        logger.info(
            "  %-12s %4d images (%d inside + %d through-doorway), %6d points -> %s",
            box.name, stats.images, selection.cameras_inside, selection.through_observers,
            stats.points, chunk_dir,
        )
        if stats.images > options.max_images_per_chunk:
            logger.warning(
                "    %d images exceeds chunk.max_images_per_chunk (%d)",
                stats.images, options.max_images_per_chunk,
            )
        if stats.points > options.target_gaussians_max:
            logger.warning(
                "    %d sparse points is above the %d target; this chunk may not fit in VRAM",
                stats.points, options.target_gaussians_max,
            )
        if stats.points_dropped_short_track:
            logger.info(
                "    %d point(s) dropped for having fewer than %d observations among the "
                "chunk's images", stats.points_dropped_short_track, options.min_track_length,
            )

    index_path = options.output_dir / INDEX_NAME
    index_path.write_text(json.dumps({
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_model": model_path.as_posix(),
        "boxes_file": options.boxes_path.as_posix(),
        "chunks": manifests,
    }, indent=2) + "\n")

    total_images = sum(int(m["num_images"]) for m in manifests)
    logger.info(
        "chunk: wrote %d chunk(s), %d failed; %d image slots across chunks "
        "(model has %d images, so chunks overlap by %.1fx)",
        len(manifests), failed, total_images, model.num_registered_images,
        total_images / model.num_registered_images if model.num_registered_images else 0.0,
    )

    orphaned = unassigned_points(model, boxes)
    if orphaned:
        logger.warning(
            "%d of %d 3D point(s) (%.1f%%) fall outside every box and are in no chunk -- "
            "that geometry will be missing from the merged result",
            orphaned, model.num_points3d, orphaned / model.num_points3d * 100,
        )

    logger.info("Index: %s", index_path)
    return EXIT_FAILED if failed else EXIT_OK


def unassigned_points(model: Model, boxes: list[BoundingBox]) -> int:
    """Points that fall in no box at all -- geometry that would be lost."""
    if not len(model.points3d):
        return 0
    covered = np.zeros(len(model.points3d), dtype=bool)
    for box in boxes:
        covered |= box.contains(model.points3d.xyz)
    return int(np.count_nonzero(~covered))


