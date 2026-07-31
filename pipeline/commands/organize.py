"""organize: lay out selected images by camera model (resolution bucket) for COLMAP."""

from __future__ import annotations

import argparse
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

VIDEO_GROUP_PREFIX = "video_"
PHOTO_GROUP_PREFIX = "photos_"

# (minimum shorter-side pixels, label), checked in order.
_VIDEO_RESOLUTION_LABELS = [
    (2100, "4k"),
    (1000, "1080p"),
    (700, "720p"),
]


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("organize", help="Organize selected images by camera model")
    return parser


def _image_size(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def _video_group_label(width: int, height: int) -> str:
    shorter_side = min(width, height)
    for threshold, label in _VIDEO_RESOLUTION_LABELS:
        if shorter_side >= threshold:
            return label
    return f"{width}x{height}"


def _photo_group_label(width: int, height: int) -> str:
    megapixels = int(width * height / 1_000_000)
    return f"{megapixels}mp"


def _iter_source_images(config: PipelineConfig) -> list[tuple[Path, str]]:
    """Return (image_path, source_type) pairs for every candidate image."""
    images: list[tuple[Path, str]] = []

    selected_root = config.paths.workdir / "selected"
    for path in sorted(selected_root.rglob("*.jpg")):
        images.append((path, "video"))

    photo_root = config.paths.input_photo_dir
    extensions = {ext.lower() for ext in config.organize.photo_extensions}
    for path in sorted(photo_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            images.append((path, "photos"))

    return images


def _group_name(source_type: str, width: int, height: int, group_by_camera_model: bool) -> str:
    if not group_by_camera_model:
        return source_type
    if source_type == "video":
        return VIDEO_GROUP_PREFIX + _video_group_label(width, height)
    return PHOTO_GROUP_PREFIX + _photo_group_label(width, height)


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    images = _iter_source_images(config)
    if not images:
        logger.warning(
            "No images found under %s or %s",
            config.paths.workdir / "selected",
            config.paths.input_photo_dir,
        )
        return 0

    images_root = config.paths.output_dir / "images"
    group_counts: dict[str, int] = defaultdict(int)
    copied, already_present, unreadable = 0, 0, 0

    for path, source_type in images:
        size = _image_size(path)
        if size is None:
            logger.warning("Could not read image, skipping: %s", path)
            unreadable += 1
            continue
        width, height = size

        group_name = _group_name(
            source_type, width, height, config.organize.group_by_camera_model
        )
        dest_dir = images_root / group_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / path.name

        if dest.exists():
            already_present += 1
        else:
            shutil.copy2(path, dest)
            copied += 1
        group_counts[group_name] += 1

    logger.info(
        "organize: %d copied, %d already present, %d unreadable",
        copied, already_present, unreadable,
    )
    for group_name, count in sorted(group_counts.items()):
        logger.info("  %-20s %d image(s)", group_name, count)
        if count < config.organize.suspicious_min_images:
            logger.warning(
                "  suspicious: group '%s' has only %d image(s) (< %d)",
                group_name, count, config.organize.suspicious_min_images,
            )

    return 0
