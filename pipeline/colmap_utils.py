"""Subprocess helpers for driving COLMAP and GLOMAP binaries."""

from __future__ import annotations

import logging
import shutil
import subprocess


class MissingBinaryError(RuntimeError):
    """Raised when a required external binary is not found on PATH."""


def require_binary(name: str) -> str:
    """Return the resolved path to `name` on PATH, or raise MissingBinaryError."""
    resolved = shutil.which(name)
    if resolved is None:
        raise MissingBinaryError(
            f"Required binary '{name}' was not found on PATH. "
            f"Install it and ensure it's on PATH before running this command."
        )
    return resolved


def run_streamed(command: list[str], logger: logging.Logger) -> int:
    """Run `command`, streaming combined stdout/stderr to `logger` line-by-line.

    Returns the process's exit code.
    """
    logger.info("$ %s", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info(line.rstrip())
    process.wait()
    return process.returncode
