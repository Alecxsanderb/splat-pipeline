"""Helpers for generating synthetic test videos with ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path


def make_synthetic_video(
    path: Path,
    size: str = "320x240",
    rate: int = 8,
    duration: float = 4.0,
    blur_enable: str | None = None,
    blur_sigma: float = 20.0,
) -> None:
    """Generate a synthetic test pattern video with ffmpeg's `testsrc` source.

    If `blur_enable` is given, it's used as the ffmpeg `enable` expression for
    a `gblur` filter (e.g. `"mod(n,2)"` to blur every other frame, or `"1"`
    to blur every frame), producing frames with deliberately reduced
    sharpness for testing frame-quality selection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    vf = f"testsrc=size={size}:rate={rate}:duration={duration}"
    filters = []
    if blur_enable is not None:
        filters.append(f"gblur=sigma={blur_sigma}:enable='{blur_enable}'")

    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", vf,
    ]
    if filters:
        command += ["-vf", ",".join(filters)]
    command.append(str(path))

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed generating {path}: {result.stderr}")
