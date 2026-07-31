"""verify: sanity-check a COLMAP reconstruction (registered images, track lengths, etc.)."""

from __future__ import annotations

import argparse
import logging

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("verify", help="Verify a COLMAP reconstruction")
    return parser


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    logger.info("verify: not yet implemented")
    return 0
