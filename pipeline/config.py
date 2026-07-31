"""Pipeline configuration: nested dataclasses with defaults, loadable from YAML."""

from __future__ import annotations

import dataclasses
import typing
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
    fps: float = 4.0
    max_dimension: int = 3840
    video_extensions: tuple[str, ...] = (".mp4", ".mov", ".hevc")
    workers: int = 4


@dataclass
class SelectConfig:
    target_min: int = 4000
    target_max: int = 5500
    blur_threshold: float = 100.0
    window_size: int = 4
    workers: int = 4


@dataclass
class SourceOverride:
    """Per-source-directory override of fps (extract) and/or window_size (select).

    `path` is matched as a directory prefix, relative to `paths.input_video_dir`
    (and mirrored under the extracted frames tree). The most specific
    (longest) matching prefix wins.
    """

    path: str = ""
    fps: float | None = None
    window_size: int | None = None


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
    overrides: list[SourceOverride] = field(default_factory=list)

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


def _list_element_type(hint: Any) -> Any:
    args = typing.get_args(hint)
    return args[0] if args else None


def _merge_dataclass(instance: Any, data: dict, context: str) -> None:
    """Recursively merge a dict of overrides into a dataclass instance in place."""
    if not isinstance(data, dict):
        raise ValueError(f"{context}: expected a mapping, got {type(data).__name__}")

    field_names = {f.name: f for f in dataclasses.fields(instance)}
    type_hints = typing.get_type_hints(type(instance))
    for key, value in data.items():
        if key not in field_names:
            raise ValueError(f"{context}: unknown config key '{key}'")

        current = getattr(instance, key)
        if dataclasses.is_dataclass(current):
            _merge_dataclass(current, value, context=f"{context}.{key}")
        elif isinstance(current, list):
            if not isinstance(value, list):
                raise ValueError(f"{context}.{key}: expected a list")
            element_type = _list_element_type(type_hints.get(key))
            if element_type is not None and dataclasses.is_dataclass(element_type):
                new_items = []
                for i, item in enumerate(value):
                    element = element_type()
                    _merge_dataclass(element, item, context=f"{context}.{key}[{i}]")
                    new_items.append(element)
                setattr(instance, key, new_items)
            else:
                setattr(instance, key, list(value))
        elif isinstance(current, Path):
            setattr(instance, key, Path(value))
        elif isinstance(current, tuple):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)


def resolve_override(
    overrides: list[SourceOverride], relative_dir: Path
) -> SourceOverride | None:
    """Return the override whose `path` is the longest matching directory prefix
    of `relative_dir`, or None if no override applies."""
    relative_parts = Path(relative_dir).parts
    best: SourceOverride | None = None
    best_length = -1
    for override in overrides:
        override_parts = Path(override.path).parts
        if len(override_parts) <= len(relative_parts) and (
            relative_parts[: len(override_parts)] == override_parts
        ):
            if len(override_parts) > best_length:
                best = override
                best_length = len(override_parts)
    return best
