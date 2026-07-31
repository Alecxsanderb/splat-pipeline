from pathlib import Path

import pytest

from pipeline.config import PipelineConfig, SourceOverride, resolve_override


def test_default_config_values():
    config = PipelineConfig.default()

    assert config.paths.input_video_dir == Path("input/video")
    assert config.paths.input_photo_dir == Path("input/photos")
    assert config.paths.workdir == Path("work")
    assert config.paths.output_dir == Path("output")

    assert config.extract.fps == 4.0
    assert config.extract.max_dimension == 3840
    assert config.extract.video_extensions == (".mp4", ".mov", ".hevc")
    assert config.extract.workers == 4

    assert config.select.target_min == 4000
    assert config.select.target_max == 5500
    assert config.select.blur_threshold == 100.0
    assert config.select.window_size == 4
    assert config.select.workers == 4

    assert config.organize.group_by_camera_model is True

    assert config.sfm.matcher == "sequential"
    assert config.sfm.camera_model == "OPENCV"

    assert config.chunk.max_images_per_chunk == 1500
    assert config.chunk.overlap == 100

    assert config.logging.level == "INFO"
    assert config.logging.log_dir == Path("logs")

    assert config.overrides == []


def test_from_yaml_partial_override(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
select:
  target_min: 4200
  target_max: 5000
logging:
  level: DEBUG
"""
    )

    config = PipelineConfig.from_yaml(yaml_path)

    assert config.select.target_min == 4200
    assert config.select.target_max == 5000
    assert config.select.blur_threshold == 100.0
    assert config.logging.level == "DEBUG"
    assert config.logging.log_dir == Path("logs")
    assert config.extract.fps == 4.0


def test_from_yaml_source_overrides(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
overrides:
  - path: video/global_sweeps
    fps: 2.0
  - path: video/room_kitchen/detail
    fps: 6.0
    window_size: 6
"""
    )

    config = PipelineConfig.from_yaml(yaml_path)

    assert len(config.overrides) == 2
    assert config.overrides[0] == SourceOverride(path="video/global_sweeps", fps=2.0)
    assert config.overrides[1] == SourceOverride(
        path="video/room_kitchen/detail", fps=6.0, window_size=6
    )


def test_resolve_override_picks_most_specific_prefix():
    overrides = [
        SourceOverride(path="video/room_kitchen", fps=3.0),
        SourceOverride(path="video/room_kitchen/detail", fps=6.0, window_size=6),
    ]

    match = resolve_override(overrides, Path("video/room_kitchen/detail"))
    assert match is not None
    assert match.fps == 6.0

    match = resolve_override(overrides, Path("video/room_kitchen"))
    assert match is not None
    assert match.fps == 3.0

    assert resolve_override(overrides, Path("video/other_room")) is None


def test_from_yaml_overrides_path_fields(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
paths:
  workdir: /tmp/some/work
"""
    )

    config = PipelineConfig.from_yaml(yaml_path)

    assert config.paths.workdir == Path("/tmp/some/work")
    assert isinstance(config.paths.workdir, Path)


def test_from_yaml_missing_file_raises(tmp_path):
    missing = tmp_path / "does-not-exist.yaml"

    with pytest.raises(FileNotFoundError):
        PipelineConfig.from_yaml(missing)


def test_from_yaml_unknown_top_level_key_raises(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("bogus_section:\n  foo: 1\n")

    with pytest.raises(ValueError, match="unknown config key"):
        PipelineConfig.from_yaml(yaml_path)


def test_from_yaml_unknown_nested_key_raises(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("select:\n  bogus_field: 1\n")

    with pytest.raises(ValueError, match="unknown config key"):
        PipelineConfig.from_yaml(yaml_path)


def test_from_yaml_empty_file_uses_defaults(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("")

    config = PipelineConfig.from_yaml(yaml_path)

    assert config == PipelineConfig.default()
