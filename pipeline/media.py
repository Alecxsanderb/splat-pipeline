"""ffmpeg/ffprobe helpers for frame extraction."""

from __future__ import annotations

import subprocess
from pathlib import Path


class FfmpegError(RuntimeError):
    """Raised when an ffmpeg or ffprobe invocation fails."""


def find_video_files(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Recursively find video files under `root` matching `extensions`."""
    if not root.is_dir():
        return []
    exts = {ext.lower() for ext in extensions}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def probe_duration_seconds(video_path: Path) -> float:
    """Return the duration of `video_path` in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise FfmpegError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    return float(result.stdout.strip())


def extract_frames(video_path: Path, out_dir: Path, clip_stem: str, fps: float) -> int:
    """Extract JPEG frames from `video_path` into `out_dir`, named `<clip_stem>_%06d.jpg`.

    Returns the number of frames written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = out_dir / f"{clip_stem}_%06d.jpg"
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        "-start_number", "0",
        str(output_pattern),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise FfmpegError(f"ffmpeg failed for {video_path}: {result.stderr.strip()}")
    return len(list(out_dir.glob(f"{clip_stem}_*.jpg")))
