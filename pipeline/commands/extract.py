"""extract: pull sampled frames from source video into the working directory."""

from __future__ import annotations

import argparse
import logging

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("extract", help="Extract frames from source video")
    return parser


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    logger.info("extract: not yet implemented")
    return 0
