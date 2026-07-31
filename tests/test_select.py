import argparse
import csv

from pipeline.commands import extract, select
from pipeline.config import (
    ExtractConfig,
    Paths,
    PipelineConfig,
    SelectConfig,
    SourceOverride,
)

from .video_fixtures import make_synthetic_video


def _config(tmp_path, extract_fps=8.0, **select_kwargs):
    return PipelineConfig(
        paths=Paths(
            input_video_dir=tmp_path / "input" / "video",
            input_photo_dir=tmp_path / "input" / "photos",
            workdir=tmp_path / "work",
            output_dir=tmp_path / "output",
        ),
        extract=ExtractConfig(fps=extract_fps, workers=2),
        select=SelectConfig(
            **{"window_size": 4, "blur_threshold": 100.0, "workers": 2, **select_kwargs}
        ),
    )


def _extract(config):
    exit_code = extract.run(argparse.Namespace(dry_run=False), config)
    assert exit_code == 0


def _read_report(report_path):
    with report_path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_select_windowing_keeps_sharpest_per_window(tmp_path):
    # Alternating sharp/blurred frames at 8fps; every other source frame is
    # heavily gblur'd (enable=mod(n,2) blurs odd source frames).
    config = _config(tmp_path, extract_fps=8.0, window_size=2, blur_threshold=1.0)
    make_synthetic_video(
        config.paths.input_video_dir / "clip.mp4",
        rate=8,
        duration=2.0,
        blur_enable="mod(n\\,2)",
    )
    _extract(config)

    exit_code = select.run(argparse.Namespace(dry_run=False), config)
    assert exit_code == 0

    report_path = config.paths.workdir / "select" / "frame_scores.csv"
    rows = {int(r["frame_index"]): r for r in _read_report(report_path)}
    assert len(rows) == 16

    for index, row in rows.items():
        expected_keep = (index % 2 == 0)
        assert row["keep"] == str(expected_keep), f"frame {index}: {row}"

    selected_dir = config.paths.workdir / "selected"
    selected_frames = sorted(selected_dir.rglob("*.jpg"))
    assert len(selected_frames) == 8
    for frame in selected_frames:
        idx = int(frame.stem.rsplit("_", 1)[1])
        assert idx % 2 == 0


def test_select_global_threshold_drops_blurry_window_best(tmp_path):
    config = _config(tmp_path, window_size=1, blur_threshold=500.0)
    make_synthetic_video(
        config.paths.input_video_dir / "sharp" / "clip.mp4", rate=8, duration=1.0
    )
    make_synthetic_video(
        config.paths.input_video_dir / "blurry" / "clip.mp4",
        rate=8,
        duration=1.0,
        blur_enable="1",
    )
    _extract(config)

    exit_code = select.run(argparse.Namespace(dry_run=False), config)
    assert exit_code == 0

    report_path = config.paths.workdir / "select" / "frame_scores.csv"
    rows = _read_report(report_path)
    sharp_rows = [r for r in rows if r["relative_dir"] == "sharp"]
    blurry_rows = [r for r in rows if r["relative_dir"] == "blurry"]
    assert len(sharp_rows) == 8
    assert len(blurry_rows) == 8

    assert all(r["is_window_best"] == "True" for r in sharp_rows + blurry_rows)
    assert all(r["keep"] == "True" for r in sharp_rows)
    assert all(r["keep"] == "False" for r in blurry_rows)
    assert all(r["meets_threshold"] == "False" for r in blurry_rows)


def test_select_dry_run_does_not_copy_files(tmp_path):
    config = _config(tmp_path, window_size=2, blur_threshold=1.0)
    make_synthetic_video(config.paths.input_video_dir / "clip.mp4", rate=8, duration=1.0)
    _extract(config)

    exit_code = select.run(argparse.Namespace(dry_run=True), config)

    assert exit_code == 0
    report_path = config.paths.workdir / "select" / "frame_scores.csv"
    assert report_path.exists()
    assert not (config.paths.workdir / "selected").exists()


def test_select_per_source_window_override(tmp_path):
    config = _config(tmp_path, window_size=2, blur_threshold=0.0)
    config.overrides = [SourceOverride(path="detail", window_size=4)]
    make_synthetic_video(
        config.paths.input_video_dir / "sweep" / "clip.mp4", rate=8, duration=1.0
    )
    make_synthetic_video(
        config.paths.input_video_dir / "detail" / "clip.mp4", rate=8, duration=1.0
    )
    _extract(config)

    exit_code = select.run(argparse.Namespace(dry_run=False), config)
    assert exit_code == 0

    report_path = config.paths.workdir / "select" / "frame_scores.csv"
    rows = _read_report(report_path)
    sweep_window_ids = {r["window_id"] for r in rows if r["relative_dir"] == "sweep"}
    detail_window_ids = {r["window_id"] for r in rows if r["relative_dir"] == "detail"}

    # 8 frames / window_size=2 (default) -> 4 windows
    # 8 frames / window_size=4 (override) -> 2 windows
    assert len(sweep_window_ids) == 4
    assert len(detail_window_ids) == 2


def test_select_missing_frames_dir_returns_error(tmp_path):
    config = _config(tmp_path)

    exit_code = select.run(argparse.Namespace(dry_run=False), config)

    assert exit_code == 1
