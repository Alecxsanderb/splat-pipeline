"""sfm: drive COLMAP feature extraction/matching and GLOMAP mapping as subprocesses."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pipeline.colmap_utils import MissingBinaryError, require_binary, run_streamed
from pipeline.commands.organize import PHOTO_GROUP_PREFIX
from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class VocabTreeNotFoundError(RuntimeError):
    """Raised when the configured vocab tree file does not exist."""


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("sfm", help="Run COLMAP + GLOMAP structure-from-motion")
    return parser


@dataclass
class SfmPaths:
    images_root: Path
    sfm_dir: Path
    database_path: Path
    sparse_output_dir: Path
    match_list_path: Path


def _sfm_paths(config: PipelineConfig) -> SfmPaths:
    sfm_dir = config.paths.workdir / "sfm"
    return SfmPaths(
        images_root=config.paths.output_dir / "images",
        sfm_dir=sfm_dir,
        database_path=sfm_dir / "database.db",
        sparse_output_dir=config.paths.output_dir / "sparse",
        match_list_path=sfm_dir / "photo_match_list.txt",
    )


def _marker_path(paths: SfmPaths, stage: str) -> Path:
    return paths.sfm_dir / f".{stage}.done"


def _build_photo_match_list(images_root: Path, match_list_path: Path) -> int:
    """Write the list of photo (non-sequential) images for vocab_tree_matcher.

    Entries are image names relative to `images_root`, matching what COLMAP's
    database stores for images under `--image_path images_root`.
    """
    entries = []
    for group_dir in sorted(images_root.glob(f"{PHOTO_GROUP_PREFIX}*")):
        if not group_dir.is_dir():
            continue
        for image_path in sorted(group_dir.iterdir()):
            if image_path.is_file():
                entries.append(f"{group_dir.name}/{image_path.name}")

    match_list_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(entries)
    match_list_path.write_text(content + "\n" if entries else "")
    return len(entries)


def _feature_extractor_command(
    colmap_bin: str, config: PipelineConfig, paths: SfmPaths, use_gpu: bool
) -> list[str]:
    return [
        colmap_bin, "feature_extractor",
        "--database_path", str(paths.database_path),
        "--image_path", str(paths.images_root),
        "--ImageReader.camera_model", config.sfm.camera_model,
        "--ImageReader.single_camera_per_folder", "1",
        "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
    ]


def _run_feature_extractor(config: PipelineConfig, paths: SfmPaths) -> bool:
    colmap_bin = require_binary("colmap")
    paths.database_path.parent.mkdir(parents=True, exist_ok=True)

    use_gpu = config.sfm.use_gpu
    exit_code = run_streamed(
        _feature_extractor_command(colmap_bin, config, paths, use_gpu), logger
    )

    if exit_code != 0 and use_gpu:
        logger.warning(
            "feature_extractor failed with use_gpu=1 (exit %d); retrying on CPU", exit_code
        )
        exit_code = run_streamed(
            _feature_extractor_command(colmap_bin, config, paths, use_gpu=False), logger
        )

    return exit_code == 0


def _run_sequential_matcher(config: PipelineConfig, paths: SfmPaths) -> bool:
    colmap_bin = require_binary("colmap")
    vocab_tree_path = config.sfm.vocab_tree_path
    if not vocab_tree_path.is_file():
        raise VocabTreeNotFoundError(f"Vocab tree file not found: {vocab_tree_path}")

    command = [
        colmap_bin, "sequential_matcher",
        "--database_path", str(paths.database_path),
        "--SequentialMatching.loop_detection", "1",
        "--SequentialMatching.vocab_tree_path", str(vocab_tree_path),
        # Matching has its own GPU switch (independent of SiftExtraction.use_gpu)
        # that defaults to on and requires an OpenGL context; force CPU so this
        # runs on headless, GPU-less hosts.
        "--SiftMatching.use_gpu", "0",
    ]
    return run_streamed(command, logger) == 0


def _run_vocab_tree_matcher(config: PipelineConfig, paths: SfmPaths) -> bool:
    colmap_bin = require_binary("colmap")
    vocab_tree_path = config.sfm.vocab_tree_path
    if not vocab_tree_path.is_file():
        raise VocabTreeNotFoundError(f"Vocab tree file not found: {vocab_tree_path}")

    photo_count = _build_photo_match_list(paths.images_root, paths.match_list_path)
    if photo_count == 0:
        logger.info("vocab_tree_matcher: no photo images found, skipping")
        return True

    command = [
        colmap_bin, "vocab_tree_matcher",
        "--database_path", str(paths.database_path),
        "--VocabTreeMatching.vocab_tree_path", str(vocab_tree_path),
        "--VocabTreeMatching.match_list_path", str(paths.match_list_path),
        "--SiftMatching.use_gpu", "0",
    ]
    return run_streamed(command, logger) == 0


def _run_glomap_mapper(config: PipelineConfig, paths: SfmPaths) -> bool:
    glomap_bin = require_binary("glomap")
    paths.sparse_output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        glomap_bin, "mapper",
        "--database_path", str(paths.database_path),
        "--image_path", str(paths.images_root),
        "--output_path", str(paths.sparse_output_dir),
    ]
    return run_streamed(command, logger) == 0


StageFn = Callable[[PipelineConfig, SfmPaths], bool]

STAGES: list[tuple[str, StageFn]] = [
    ("feature_extractor", _run_feature_extractor),
    ("sequential_matcher", _run_sequential_matcher),
    ("vocab_tree_matcher", _run_vocab_tree_matcher),
    ("glomap_mapper", _run_glomap_mapper),
]


def _write_timing_summary(rows: list[tuple[str, str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "status", "duration_seconds"])
        writer.writerows(rows)


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    paths = _sfm_paths(config)
    paths.sfm_dir.mkdir(parents=True, exist_ok=True)

    timing_rows: list[tuple[str, str, float]] = []
    failed = False

    for stage_name, stage_fn in STAGES:
        marker = _marker_path(paths, stage_name)
        if marker.exists():
            logger.info("skip %s (already completed)", stage_name)
            timing_rows.append((stage_name, "skipped", 0.0))
            continue

        logger.info("running %s", stage_name)
        start = time.perf_counter()
        try:
            success = stage_fn(config, paths)
        except (MissingBinaryError, VocabTreeNotFoundError) as exc:
            logger.error("%s", exc)
            success = False
        duration = time.perf_counter() - start

        if success:
            marker.write_text(f"{duration:.2f}\n")
            timing_rows.append((stage_name, "completed", round(duration, 2)))
            logger.info("%s completed in %.2fs", stage_name, duration)
        else:
            timing_rows.append((stage_name, "failed", round(duration, 2)))
            logger.error("%s failed after %.2fs", stage_name, duration)
            failed = True
            break

    timing_path = paths.sfm_dir / "timing_summary.csv"
    _write_timing_summary(timing_rows, timing_path)

    logger.info("sfm: timing summary (%s)", timing_path)
    for stage_name, status, duration in timing_rows:
        logger.info("  %-20s %-10s %8.2fs", stage_name, status, duration)

    return 1 if failed else 0
