"""Pipeline configuration: nested dataclasses with defaults, loadable from YAML."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Paths:
    input_video_dir: Path = Path("input/video")
    input_photo_dir: Path = Path("input/photos")
    workdir: Path = Path("work")
    output_dir: Path = Path("output")


@dataclass
class ExtractConfig:
    fps: float = 2.0
    max_dimension: int = 3840
    video_extensions: tuple[str, ...] = (".mp4", ".mov", ".hevc")


@dataclass
class SelectConfig:
    target_min: int = 4000
    target_max: int = 5500
    blur_threshold: float = 100.0


@dataclass
class OrganizeConfig:
    group_by_camera_model: bool = True


@dataclass
class SfmConfig:
    matcher: str = "sequential"
    camera_model: str = "OPENCV"


@dataclass
class ChunkConfig:
    max_images_per_chunk: int = 1500
    overlap: int = 100


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: Path = Path("logs")


@dataclass
class PipelineConfig:
    paths: Paths = field(default_factory=Paths)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    organize: OrganizeConfig = field(default_factory=OrganizeConfig)
    sfm: SfmConfig = field(default_factory=SfmConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def default(cls) -> PipelineConfig:
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r") as f:
            data = yaml.safe_load(f) or {}

        config = cls.default()
        _merge_dataclass(config, data, context=path.as_posix())
        return config


def _merge_dataclass(instance: Any, data: dict, context: str) -> None:
    """Recursively merge a dict of overrides into a dataclass instance in place."""
    if not isinstance(data, dict):
        raise ValueError(f"{context}: expected a mapping, got {type(data).__name__}")

    field_names = {f.name: f for f in dataclasses.fields(instance)}
    for key, value in data.items():
        if key not in field_names:
            raise ValueError(f"{context}: unknown config key '{key}'")

        current = getattr(instance, key)
        if dataclasses.is_dataclass(current):
            _merge_dataclass(current, value, context=f"{context}.{key}")
        elif isinstance(current, Path):
            setattr(instance, key, Path(value))
        elif isinstance(current, tuple):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)
