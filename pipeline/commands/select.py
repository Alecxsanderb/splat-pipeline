"""select: choose a well-distributed, sharp subset of candidate images."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pipeline.config import PipelineConfig, resolve_override
from pipeline.sharpness import variance_of_laplacian

logger = logging.getLogger(__name__)

FRAME_NAME_RE = re.compile(r"^(?P<clip>.+)_(?P<index>\d{6})$")


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("select", help="Select a subset of images for reconstruction")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score frames and write the report without copying selected frames",
    )
    return parser


@dataclass
class Frame:
    path: Path
    relative_dir: Path
    clip: str
    index: int
    score: float = 0.0
    window_id: int = -1
    is_window_best: bool = False
    meets_threshold: bool = False

    @property
    def keep(self) -> bool:
        return self.is_window_best and self.meets_threshold


def _discover_frames(frames_root: Path) -> list[Frame]:
    frames = []
    for path in sorted(frames_root.rglob("*.jpg")):
        match = FRAME_NAME_RE.match(path.stem)
        if not match:
            logger.warning("Skipping unrecognized frame filename: %s", path)
            continue
        frames.append(
            Frame(
                path=path,
                relative_dir=path.parent.relative_to(frames_root),
                clip=match.group("clip"),
                index=int(match.group("index")),
            )
        )
    return frames


def _group_by_clip(frames: list[Frame]) -> dict[tuple[Path, str], list[Frame]]:
    groups: dict[tuple[Path, str], list[Frame]] = {}
    for frame in frames:
        groups.setdefault((frame.relative_dir, frame.clip), []).append(frame)
    for group_frames in groups.values():
        group_frames.sort(key=lambda f: f.index)
    return groups


def _score_frames(frames: list[Frame], workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = pool.map(lambda frame: variance_of_laplacian(frame.path), frames)
        for frame, score in zip(frames, scores, strict=True):
            frame.score = score


def _apply_windowed_selection(
    groups: dict[tuple[Path, str], list[Frame]], config: PipelineConfig
) -> None:
    for (relative_dir, _clip), group_frames in groups.items():
        override = resolve_override(config.overrides, relative_dir)
        window = (
            override.window_size
            if override and override.window_size is not None
            else config.select.window_size
        )
        window = max(1, window)

        for start in range(0, len(group_frames), window):
            chunk = group_frames[start : start + window]
            window_id = start // window
            best = max(chunk, key=lambda f: f.score)
            for frame in chunk:
                frame.window_id = window_id
                frame.is_window_best = frame is best
                frame.meets_threshold = frame.score >= config.select.blur_threshold


def _write_report(frames: list[Frame], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "relative_dir",
                "clip",
                "frame_index",
                "path",
                "sharpness_score",
                "window_id",
                "is_window_best",
                "meets_threshold",
                "keep",
            ]
        )
        for frame in sorted(frames, key=lambda f: (str(f.relative_dir), f.clip, f.index)):
            writer.writerow(
                [
                    frame.relative_dir.as_posix(),
                    frame.clip,
                    frame.index,
                    frame.path.as_posix(),
                    f"{frame.score:.4f}",
                    frame.window_id,
                    frame.is_window_best,
                    frame.meets_threshold,
                    frame.keep,
                ]
            )


def _copy_selected(frames: list[Frame], selected_root: Path) -> int:
    copied = 0
    for frame in frames:
        if not frame.keep:
            continue
        dest_dir = selected_root / frame.relative_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / frame.path.name
        if not dest.exists():
            shutil.copy2(frame.path, dest)
            copied += 1
    return copied


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    frames_root = config.paths.workdir / "frames"
    if not frames_root.is_dir():
        logger.error("No extracted frames found at %s; run `splat extract` first", frames_root)
        return 1

    frames = _discover_frames(frames_root)
    if not frames:
        logger.warning("No frames found under %s", frames_root)
        return 0

    _score_frames(frames, config.select.workers)
    groups = _group_by_clip(frames)
    _apply_windowed_selection(groups, config)

    report_path = config.paths.workdir / "select" / "frame_scores.csv"
    _write_report(frames, report_path)

    kept = sum(1 for frame in frames if frame.keep)
    logger.info(
        "select: %d frame(s) scored, %d kept, %d dropped (report: %s)",
        len(frames), kept, len(frames) - kept, report_path,
    )

    if getattr(args, "dry_run", False):
        logger.info("[dry-run] no frames copied")
        return 0

    selected_root = config.paths.workdir / "selected"
    copied = _copy_selected(frames, selected_root)
    logger.info("select: copied %d newly selected frame(s) to %s", copied, selected_root)
    return 0
