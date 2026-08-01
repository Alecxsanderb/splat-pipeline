"""Argparse entry point for the `splat` command."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from pipeline.commands import chunk, extract, merge, organize, select, sfm, verify
from pipeline.config import PipelineConfig
from pipeline.logging_setup import configure_logging

logger = logging.getLogger(__name__)

COMMAND_MODULES = [extract, select, organize, sfm, verify, chunk, merge]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splat",
        description="Process video and photos into a COLMAP reconstruction for Gaussian Splatting.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override the console logging level from the config",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    for module in COMMAND_MODULES:
        module.add_parser(subparsers)

    return parser


def _load_config(args: argparse.Namespace) -> PipelineConfig:
    if args.config:
        return PipelineConfig.from_yaml(args.config)
    return PipelineConfig.default()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = _load_config(args)
    if args.log_level:
        config.logging.level = args.log_level

    log_file = configure_logging(config.logging)
    logger.info("Logging to %s", log_file)

    for module in COMMAND_MODULES:
        if module.__name__.rsplit(".", 1)[-1] == args.command:
            return module.run(args, config)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
