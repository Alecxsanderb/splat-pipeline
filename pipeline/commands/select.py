"""select: choose a well-distributed, sharp subset of candidate images."""

from __future__ import annotations

import argparse
import logging

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("select", help="Select a subset of images for reconstruction")
    return parser


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    logger.info("select: not yet implemented")
    return 0
