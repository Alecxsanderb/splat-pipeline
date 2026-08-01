import argparse

import cv2
import numpy as np

from pipeline.commands import organize
from pipeline.config import OrganizeConfig, Paths, PipelineConfig


def _write_image(path, width, height):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _config(tmp_path, **organize_kwargs):
    return PipelineConfig(
        paths=Paths(
            input_video_dir=tmp_path / "input" / "video",
            input_photo_dir=tmp_path / "input" / "photos",
            workdir=tmp_path / "work",
            output_dir=tmp_path / "output",
        ),
        organize=OrganizeConfig(**{"suspicious_min_images": 2, **organize_kwargs}),
    )


def test_organize_groups_by_source_and_resolution(tmp_path):
    config = _config(tmp_path)

    for i in range(3):
        _write_image(
            config.paths.workdir / "selected" / "room1" / f"clip_{i:06d}.jpg", 3840, 2160
        )
    for i in range(2):
        _write_image(config.paths.input_photo_dir / f"IMG_{i:04d}.jpg", 8064, 6048)

    exit_code = organize.run(argparse.Namespace(), config)

    assert exit_code == 0
    images_root = config.paths.output_dir / "images"
    assert sorted(p.name for p in images_root.iterdir()) == ["photos_48mp", "video_4k"]
    assert len(list((images_root / "video_4k").glob("*.jpg"))) == 3
    assert len(list((images_root / "photos_48mp").glob("*.jpg"))) == 2


def test_organize_flags_suspicious_group(tmp_path, caplog):
    config = _config(tmp_path, suspicious_min_images=5)
    _write_image(config.paths.workdir / "selected" / "room1" / "clip_000000.jpg", 1920, 1080)
    for i in range(2):
        _write_image(config.paths.input_photo_dir / f"IMG_{i:04d}.jpg", 4032, 3024)

    with caplog.at_level("WARNING"):
        exit_code = organize.run(argparse.Namespace(), config)

    assert exit_code == 0
    messages = [r.message for r in caplog.records]
    assert any("suspicious" in m and "video_1080p" in m for m in messages)
    assert any("suspicious" in m and "photos_12mp" in m for m in messages)


def test_organize_is_resumable_skips_existing_copies(tmp_path):
    config = _config(tmp_path)
    frame_path = config.paths.workdir / "selected" / "room1" / "clip_000000.jpg"
    _write_image(frame_path, 1920, 1080)

    organize.run(argparse.Namespace(), config)
    dest = config.paths.output_dir / "images" / "video_1080p" / "clip_000000.jpg"
    first_mtime = dest.stat().st_mtime_ns

    organize.run(argparse.Namespace(), config)
    assert dest.stat().st_mtime_ns == first_mtime


def test_organize_flat_layout_when_grouping_disabled(tmp_path):
    config = _config(tmp_path, group_by_camera_model=False)
    _write_image(config.paths.workdir / "selected" / "room1" / "clip_000000.jpg", 1920, 1080)
    _write_image(config.paths.input_photo_dir / "IMG_0000.jpg", 4032, 3024)

    exit_code = organize.run(argparse.Namespace(), config)

    assert exit_code == 0
    images_root = config.paths.output_dir / "images"
    assert sorted(p.name for p in images_root.iterdir()) == ["photos", "video"]


def test_organize_no_images_found_returns_ok(tmp_path):
    config = _config(tmp_path)

    exit_code = organize.run(argparse.Namespace(), config)

    assert exit_code == 0
    assert not (config.paths.output_dir / "images").exists()


def test_organize_skips_unreadable_image(tmp_path, caplog):
    config = _config(tmp_path)
    bad_path = config.paths.input_photo_dir / "corrupt.jpg"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("not actually a jpeg")
    _write_image(config.paths.input_photo_dir / "IMG_0000.jpg", 4032, 3024)

    with caplog.at_level("WARNING"):
        exit_code = organize.run(argparse.Namespace(), config)

    assert exit_code == 0
    assert any("Could not read image" in r.message for r in caplog.records)
    photo_group = config.paths.output_dir / "images" / "photos_12mp"
    assert len(list(photo_group.glob("*.jpg"))) == 1
