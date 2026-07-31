"""chunk: split a large reconstruction into overlapping spatial chunks."""

from __future__ import annotations

import argparse
import logging

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("chunk", help="Split reconstruction into overlapping chunks")
    return parser


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    logger.info("chunk: not yet implemented")
    return 0
