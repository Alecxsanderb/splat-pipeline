"""extract: pull sampled frames from source video into the working directory."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pipeline.config import PipelineConfig, resolve_override
from pipeline.media import FfmpegError, extract_frames, find_video_files, probe_duration_seconds

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("extract", help="Extract frames from source video")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate frame counts per clip (via ffprobe) without running ffmpeg",
    )
    return parser


@dataclass
class ClipJob:
    video_path: Path
    relative_dir: Path
    out_dir: Path
    clip_stem: str
    fps: float


def _plan_jobs(config: PipelineConfig) -> list[ClipJob]:
    input_dir = config.paths.input_video_dir
    frames_root = config.paths.workdir / "frames"
    jobs = []
    for video_path in find_video_files(input_dir, config.extract.video_extensions):
        relative_dir = video_path.parent.relative_to(input_dir)
        override = resolve_override(config.overrides, relative_dir)
        fps = override.fps if override and override.fps is not None else config.extract.fps
        jobs.append(
            ClipJob(
                video_path=video_path,
                relative_dir=relative_dir,
                out_dir=frames_root / relative_dir,
                clip_stem=video_path.stem,
                fps=fps,
            )
        )
    return jobs


def _marker_path(job: ClipJob) -> Path:
    return job.out_dir / f".{job.clip_stem}.done"


def _extract_one(job: ClipJob) -> tuple[int, bool]:
    """Returns (frame_count, was_skipped)."""
    marker = _marker_path(job)
    if marker.exists():
        existing = len(list(job.out_dir.glob(f"{job.clip_stem}_*.jpg")))
        return existing, True

    count = extract_frames(job.video_path, job.out_dir, job.clip_stem, job.fps)
    marker.write_text(str(count))
    return count, False


def _dry_run(jobs: list[ClipJob]) -> int:
    total_estimated = 0
    for job in jobs:
        try:
            duration = probe_duration_seconds(job.video_path)
        except FfmpegError as exc:
            logger.error("%s", exc)
            continue
        estimated = int(duration * job.fps)
        total_estimated += estimated
        logger.info(
            "[dry-run] %s: fps=%s duration=%.2fs -> ~%d frames",
            job.video_path, job.fps, duration, estimated,
        )
    logger.info("[dry-run] %d clip(s), ~%d frames total", len(jobs), total_estimated)
    return 0


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    jobs = _plan_jobs(config)
    if not jobs:
        logger.warning("No video files found under %s", config.paths.input_video_dir)
        return 0

    if getattr(args, "dry_run", False):
        return _dry_run(jobs)

    extracted, skipped, failed = 0, 0, 0
    with ThreadPoolExecutor(max_workers=config.extract.workers) as pool:
        future_to_job = {pool.submit(_extract_one, job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                count, was_skipped = future.result()
            except FfmpegError as exc:
                logger.error("%s", exc)
                failed += 1
                continue

            if was_skipped:
                skipped += 1
                logger.info("skip %s (already extracted, %d frames)", job.video_path, count)
            else:
                extracted += 1
                logger.info("extracted %s -> %d frames", job.video_path, count)

    logger.info("extract: %d extracted, %d skipped, %d failed", extracted, skipped, failed)
    return 1 if failed else 0
