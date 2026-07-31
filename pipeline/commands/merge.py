"""merge: recombine chunked reconstructions/splats into a single result."""

from __future__ import annotations

import argparse
import logging

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("merge", help="Merge chunked results back together")
    return parser


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    logger.info("merge: not yet implemented")
    return 0
