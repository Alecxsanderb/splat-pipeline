"""sfm: run COLMAP structure-from-motion on the organized image set."""

from __future__ import annotations

import argparse
import logging

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("sfm", help="Run COLMAP structure-from-motion")
    return parser


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    logger.info("sfm: not yet implemented")
    return 0
