"""merge: crop trained per-chunk splats to their boxes and concatenate them.

Chunks deliberately overlap -- an image looking through a doorway belongs to
both rooms -- so the trained splats overlap too, and concatenating them raw
would leave doubled, z-fighting geometry at every shared wall. Each chunk is
therefore cropped back to its manifest's box (plus a margin, so seams are not
razor-thin) before being merged.

Exit codes:
    0  merged output written
    1  at least one chunk could not be merged
    3  nothing to merge (no chunks, or no trained splats found)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pipeline.boxes import BoundingBox, BoxError
from pipeline.commands.chunk import MANIFEST_NAME
from pipeline.config import PipelineConfig
from pipeline.ply import GaussianCloud, PlyError, concatenate, read_ply, write_ply

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOTHING_TO_DO = 3


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("merge", help="Merge trained chunk splats into one PLY")
    # Every argument must stay optional; `splat merge` is exercised bare by the
    # CLI tests.
    parser.add_argument("--chunks-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ply-name", type=str, default=None,
                        help="Filename of the trained splat inside each chunk directory")
    parser.add_argument("--crop-margin", type=float, default=None,
                        help="Grow each manifest box by this much before cropping")
    parser.add_argument("--no-crop", action="store_true",
                        help="Concatenate without cropping (will duplicate overlap geometry)")
    return parser


@dataclass(frozen=True)
class MergeOptions:
    chunks_dir: Path
    output_path: Path
    ply_name: str
    crop_margin: float
    crop: bool


def _resolve_options(args: argparse.Namespace, config: PipelineConfig) -> MergeOptions:
    cfg = config.merge

    def pick(name: str, fallback):
        value = getattr(args, name, None)
        return fallback if value is None else value

    return MergeOptions(
        chunks_dir=pick("chunks_dir", config.paths.output_dir / "chunks"),
        output_path=pick("output", config.paths.output_dir / cfg.output_name),
        ply_name=pick("ply_name", cfg.ply_name),
        crop_margin=pick("crop_margin", cfg.crop_margin),
        crop=not getattr(args, "no_crop", False),
    )


@dataclass
class ChunkInput:
    name: str
    manifest_path: Path
    ply_path: Path
    box: BoundingBox


def find_chunk_inputs(chunks_dir: Path, ply_name: str) -> tuple[list[ChunkInput], list[str]]:
    """Pair each chunk manifest with its trained splat.

    3DGS writes its output nested (``point_cloud/iteration_30000/point_cloud.ply``),
    so the named file is searched for recursively and the deepest, most
    recently modified match wins -- that is the latest training iteration.
    """
    found: list[ChunkInput] = []
    problems: list[str] = []

    if not chunks_dir.is_dir():
        return found, [f"chunks directory not found: {chunks_dir}"]

    for manifest_path in sorted(chunks_dir.glob(f"*/{MANIFEST_NAME}")):
        chunk_dir = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text())
            box = BoundingBox.from_dict(manifest["box"])
        except (OSError, ValueError, KeyError, BoxError) as exc:
            problems.append(f"{manifest_path}: unreadable manifest ({exc})")
            continue

        candidates = sorted(chunk_dir.rglob(ply_name))
        if not candidates:
            problems.append(
                f"{chunk_dir.name}: no {ply_name} found -- has this chunk been trained yet?"
            )
            continue
        # Latest training iteration: newest mtime, deepest path as a tiebreak.
        ply_path = max(candidates, key=lambda p: (p.stat().st_mtime, len(p.parts)))
        if len(candidates) > 1:
            logger.info(
                "  %s: %d candidate splats, using %s", chunk_dir.name, len(candidates), ply_path
            )
        found.append(
            ChunkInput(
                name=str(manifest.get("name", chunk_dir.name)),
                manifest_path=manifest_path,
                ply_path=ply_path,
                box=box,
            )
        )

    return found, problems


def crop_to_box(cloud: GaussianCloud, box: BoundingBox, margin: float) -> GaussianCloud:
    """Keep only Gaussians whose centre lies inside the box grown by `margin`."""
    region = box.expanded(margin) if margin else box
    return cloud.select(region.contains(cloud.xyz))


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    options = _resolve_options(args, config)

    inputs, problems = find_chunk_inputs(options.chunks_dir, options.ply_name)
    for problem in problems:
        logger.warning("%s", problem)

    if not inputs:
        logger.error(
            "No trained chunk splats found under %s (looking for %s in each chunk directory)",
            options.chunks_dir, options.ply_name,
        )
        return EXIT_NOTHING_TO_DO

    logger.info(
        "Merging %d chunk(s) from %s%s",
        len(inputs), options.chunks_dir,
        f" (crop margin {options.crop_margin})" if options.crop else " (cropping disabled)",
    )

    clouds: list[GaussianCloud] = []
    total_before = 0
    failed = 0

    for chunk in inputs:
        try:
            cloud = read_ply(chunk.ply_path)
        except PlyError as exc:
            logger.error("  %-12s could not be read: %s", chunk.name, exc)
            failed += 1
            continue

        before = len(cloud)
        total_before += before
        if options.crop:
            try:
                cloud = crop_to_box(cloud, chunk.box, options.crop_margin)
            except BoxError as exc:
                logger.error("  %-12s could not be cropped: %s", chunk.name, exc)
                failed += 1
                continue

        after = len(cloud)
        removed = before - after
        logger.info(
            "  %-12s %9d -> %9d Gaussians (%d cropped away, %.1f%%)",
            chunk.name, before, after, removed, (removed / before * 100) if before else 0.0,
        )
        if after == 0:
            logger.warning(
                "    %s: cropping removed every Gaussian -- is the splat in the same "
                "coordinate frame as the box it was chunked from?", chunk.name,
            )
        clouds.append(cloud)

    if not clouds:
        logger.error("No chunk splats could be read")
        return EXIT_FAILED

    try:
        merged = concatenate(clouds)
    except PlyError as exc:
        logger.error("%s", exc)
        return EXIT_FAILED

    write_ply(
        options.output_path,
        merged,
        comments=[f"merged from {len(clouds)} chunk(s) by splat-pipeline"],
    )

    logger.info(
        "merge: %d Gaussians in, %d out (%d removed by cropping, %.1f%%)",
        total_before, len(merged), total_before - len(merged),
        ((total_before - len(merged)) / total_before * 100) if total_before else 0.0,
    )
    logger.info("Properties preserved (%d): %s",
                len(merged.properties), ", ".join(merged.property_names))
    logger.info("Wrote %s", options.output_path)
    return EXIT_FAILED if failed else EXIT_OK
