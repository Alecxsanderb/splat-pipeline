"""organize: lay out selected images by camera model for COLMAP."""

from __future__ import annotations

import argparse
import logging

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("organize", help="Organize selected images by camera model")
    return parser


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    logger.info("organize: not yet implemented")
    return 0
