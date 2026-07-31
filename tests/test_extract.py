import argparse
import time

from pipeline.commands import extract
from pipeline.config import ExtractConfig, Paths, PipelineConfig, SourceOverride

from .video_fixtures import make_synthetic_video


def _config(tmp_path, **extract_kwargs):
    return PipelineConfig(
        paths=Paths(
            input_video_dir=tmp_path / "input" / "video",
            input_photo_dir=tmp_path / "input" / "photos",
            workdir=tmp_path / "work",
            output_dir=tmp_path / "output",
        ),
        extract=ExtractConfig(**{"fps": 4.0, "workers": 2, **extract_kwargs}),
    )


def test_extract_basic_creates_expected_frame_count(tmp_path):
    config = _config(tmp_path)
    video_path = config.paths.input_video_dir / "clip.mp4"
    make_synthetic_video(video_path, rate=8, duration=2.0)

    exit_code = extract.run(argparse.Namespace(dry_run=False), config)

    assert exit_code == 0
    out_dir = config.paths.workdir / "frames"
    frames = sorted(out_dir.glob("clip_*.jpg"))
    assert len(frames) == 8  # 2s @ fps=4
    assert (out_dir / ".clip.done").exists()


def test_extract_is_resumable_skips_existing(tmp_path, caplog):
    config = _config(tmp_path)
    video_path = config.paths.input_video_dir / "clip.mp4"
    make_synthetic_video(video_path, rate=8, duration=1.0)

    extract.run(argparse.Namespace(dry_run=False), config)
    out_dir = config.paths.workdir / "frames"
    frame_file = sorted(out_dir.glob("clip_*.jpg"))[0]
    first_mtime = frame_file.stat().st_mtime_ns

    time.sleep(0.05)
    with caplog.at_level("INFO"):
        exit_code = extract.run(argparse.Namespace(dry_run=False), config)

    assert exit_code == 0
    assert frame_file.stat().st_mtime_ns == first_mtime
    assert any("skip" in record.message for record in caplog.records)


def test_extract_dry_run_reports_without_writing_files(tmp_path, caplog):
    config = _config(tmp_path, fps=4.0)
    video_path = config.paths.input_video_dir / "clip.mp4"
    make_synthetic_video(video_path, rate=8, duration=2.0)

    with caplog.at_level("INFO"):
        exit_code = extract.run(argparse.Namespace(dry_run=True), config)

    assert exit_code == 0
    assert not (config.paths.workdir / "frames").exists()
    assert any("dry-run" in record.message for record in caplog.records)
    assert any("~8 frames" in record.message for record in caplog.records)


def test_extract_per_source_fps_override(tmp_path):
    config = _config(tmp_path, fps=2.0)
    config.overrides = [SourceOverride(path="room_b", fps=8.0)]

    make_synthetic_video(
        config.paths.input_video_dir / "room_a" / "clip.mp4", rate=8, duration=2.0
    )
    make_synthetic_video(
        config.paths.input_video_dir / "room_b" / "clip.mp4", rate=8, duration=2.0
    )

    exit_code = extract.run(argparse.Namespace(dry_run=False), config)

    assert exit_code == 0
    frames_root = config.paths.workdir / "frames"
    room_a_frames = list((frames_root / "room_a").glob("clip_*.jpg"))
    room_b_frames = list((frames_root / "room_b").glob("clip_*.jpg"))
    assert len(room_a_frames) == 4  # 2s @ fps=2.0 (config default)
    assert len(room_b_frames) == 16  # 2s @ fps=8.0 (override)


def test_extract_reports_failure_for_invalid_video(tmp_path, caplog):
    config = _config(tmp_path)
    bad_video = config.paths.input_video_dir / "not_really_a_video.mp4"
    bad_video.parent.mkdir(parents=True, exist_ok=True)
    bad_video.write_text("this is not a video file")

    with caplog.at_level("ERROR"):
        exit_code = extract.run(argparse.Namespace(dry_run=False), config)

    assert exit_code == 1
    assert any("ffmpeg failed" in record.message for record in caplog.records)


def test_extract_multiple_clips_in_parallel(tmp_path):
    config = _config(tmp_path, workers=3)
    for i in range(3):
        make_synthetic_video(
            config.paths.input_video_dir / f"clip{i}.mp4", rate=8, duration=1.0
        )

    exit_code = extract.run(argparse.Namespace(dry_run=False), config)

    assert exit_code == 0
    frames_root = config.paths.workdir / "frames"
    for i in range(3):
        assert len(list(frames_root.glob(f"clip{i}_*.jpg"))) == 4
