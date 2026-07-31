"""Validates the binary reader against real COLMAP.

A reader tested only against our own writer proves nothing if both share the
same misunderstanding of the format, so these tests put the actual `colmap`
binary on one side of every comparison.

`model_converter` is used rather than `mapper` deliberately: it reads and
writes models directly, so nothing here depends on a reconstruction
successfully initialising (which is unreliable on synthetic imagery, where
planar or degenerate scenes routinely fail).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess

import numpy as np
import pytest

from pipeline.colmap_model import CAMERA_MODELS_BY_NAME, read_model

from .colmap_fixtures import (
    FixtureCamera,
    FixtureImage,
    FixtureModel,
    FixturePoint3D,
    build_healthy_model,
    make_consistent,
    write_model_bin,
    write_model_txt,
)

COLMAP_PATH = shutil.which("colmap")

pytestmark = [
    pytest.mark.colmap_integration,
    pytest.mark.skipif(COLMAP_PATH is None, reason="colmap binary not found on PATH"),
]


def _convert(input_path, output_path, output_type: str) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            COLMAP_PATH, "model_converter",
            "--input_path", str(input_path),
            "--output_path", str(output_path),
            "--output_type", output_type,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"colmap model_converter -> {output_type} failed:\n{result.stdout}\n{result.stderr}"
    )


def _all_camera_models_fixture() -> FixtureModel:
    """One camera per COLMAP camera model, so real COLMAP pins the whole id table."""
    cameras = [
        FixtureCamera(camera_id=i + 1, model=name, width=1920, height=1080)
        for i, name in enumerate(sorted(CAMERA_MODELS_BY_NAME))
    ]
    images = [
        FixtureImage(
            image_id=i + 1,
            name=f"video_test/frame_{i:06d}.jpg",
            camera_id=camera.camera_id,
            qvec=(1.0, 0.0, 0.0, 0.0),
            tvec=(float(i), 0.5, 0.25),
            points2d=[(100.5, 200.25, 1), (300.0, 400.0, None)],
        )
        for i, camera in enumerate(cameras)
    ]
    points = [
        FixturePoint3D(
            point3d_id=1,
            xyz=(1.5, 2.25, 3.0),
            rgb=(10, 20, 30),
            error=0.5,
            track=[(img.image_id, 0) for img in images],
        )
    ]
    return make_consistent(FixtureModel(cameras=cameras, images=images, points3d=points))


def test_reader_agrees_with_colmap_binary_writer(tmp_path):
    """TXT (ours) -> colmap model_converter -> BIN -> our reader.

    Proves the reader matches COLMAP's own binary *writer*, and pins the entire
    camera-model id table, since COLMAP resolves the model names itself.
    """
    fixture = _all_camera_models_fixture()
    write_model_txt(tmp_path / "txt", fixture)
    _convert(tmp_path / "txt", tmp_path / "bin", "BIN")

    model = read_model(tmp_path / "bin")

    # Compare as dicts keyed by id: COLMAP stores records in unordered_map, so
    # on-disk order is hash order and any sequence comparison would be flaky.
    assert len(model.cameras) == len(fixture.cameras)
    for expected in fixture.cameras:
        camera = model.cameras[expected.camera_id]
        assert camera.model_name == expected.model_name
        assert camera.model_id == CAMERA_MODELS_BY_NAME[expected.model_name].model_id
        assert (camera.width, camera.height) == (expected.width, expected.height)
        assert camera.params == expected.resolved_params()

    assert len(model.images) == len(fixture.images)
    for expected in fixture.images:
        image = model.images[expected.image_id]
        assert image.name == expected.name
        assert image.camera_id == expected.camera_id
        assert image.num_points2d == len(expected.points2d)
        np.testing.assert_allclose(image.qvec, expected.qvec, atol=1e-12)
        np.testing.assert_allclose(image.tvec, expected.tvec, atol=1e-12)

    assert model.num_points3d == 1
    point = model.points3d[1]
    assert tuple(point.xyz) == (1.5, 2.25, 3.0)
    assert tuple(point.rgb) == (10, 20, 30)
    assert point.error == 0.5
    assert point.track_length == len(fixture.images)


def test_txt_minus_one_maps_to_binary_sentinel(tmp_path):
    """The TXT format spells 'no 3D point' as -1, the BIN format as 2**64-1."""
    fixture = make_consistent(
        FixtureModel(
            cameras=[FixtureCamera(camera_id=1, model="PINHOLE")],
            images=[
                FixtureImage(
                    image_id=1,
                    name="g/a.jpg",
                    points2d=[(1.0, 2.0, 1), (3.0, 4.0, None), (5.0, 6.0, None)],
                ),
                FixtureImage(image_id=2, name="g/b.jpg", points2d=[(7.0, 8.0, 1)]),
            ],
            points3d=[FixturePoint3D(1, (0.5, 0.5, 0.5), track=[(1, 0), (2, 0)])],
        )
    )
    write_model_txt(tmp_path / "txt", fixture)
    _convert(tmp_path / "txt", tmp_path / "bin", "BIN")

    image = read_model(tmp_path / "bin").images[1]

    assert image.num_valid_points3d == 1
    assert image.valid_point3d_ids().tolist() == [1]


def test_colmap_can_read_our_binary_writer(tmp_path):
    """BIN (ours) -> colmap model_converter -> TXT -> parsed here.

    Proves the fixture writer is correct, which is what makes every round-trip
    test in test_colmap_model.py meaningful.
    """
    fixture = build_healthy_model(num_images=5, num_points=30, error=0.5)
    write_model_bin(tmp_path / "bin", fixture)
    _convert(tmp_path / "bin", tmp_path / "txt", "TXT")

    cameras = {}
    for line in (tmp_path / "txt" / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        cameras[int(parts[0])] = (parts[1], int(parts[2]), int(parts[3]))

    expected_camera = fixture.cameras[0]
    assert cameras[expected_camera.camera_id] == (
        expected_camera.model_name, expected_camera.width, expected_camera.height,
    )

    names = {}
    lines = [
        line for line in (tmp_path / "txt" / "images.txt").read_text().splitlines()
        if not line.startswith("#")
    ]
    # COLMAP writes exactly two lines per image: pose line, then point2D line.
    for pose_line in lines[::2]:
        if not pose_line.strip():
            continue
        parts = pose_line.split()
        names[int(parts[0])] = parts[9]

    assert names == {img.image_id: img.name for img in fixture.images}


def test_round_trip_through_colmap_preserves_model(tmp_path):
    """BIN (ours) -> TXT (colmap) -> BIN (colmap) -> our reader."""
    fixture = build_healthy_model(num_images=5, num_points=30, error=0.5)
    write_model_bin(tmp_path / "bin", fixture)
    _convert(tmp_path / "bin", tmp_path / "txt", "TXT")
    _convert(tmp_path / "txt", tmp_path / "bin2", "BIN")

    model = read_model(tmp_path / "bin2")

    assert model.num_registered_images == len(fixture.images)
    assert model.num_points3d == len(fixture.points3d)
    assert model.num_observations == sum(len(p.track) for p in fixture.points3d)
    assert {img.name for img in model.images.values()} == {img.name for img in fixture.images}


def test_model_analyzer_agrees_with_our_statistics(tmp_path):
    """`colmap model_analyzer` as an independent oracle for our aggregates."""
    fixture = build_healthy_model(num_images=8, num_points=60, error=0.5)
    write_model_bin(tmp_path / "bin", fixture)
    model = read_model(tmp_path / "bin")

    result = subprocess.run(
        [COLMAP_PATH, "model_analyzer", "--path", str(tmp_path / "bin")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    logging.getLogger(__name__).info("model_analyzer said:\n%s", output)

    # Parse defensively: assert only on the keys that matched, so a COLMAP
    # wording change weakens coverage rather than reddening the suite.
    found: dict[str, float] = {}
    patterns = {
        "images": r"Registered images:\s*([0-9]+)",
        "points": r"Points:\s*([0-9]+)",
        "observations": r"Observations:\s*([0-9]+)",
        "mean_track_length": r"Mean track length:\s*([0-9.]+)",
        "mean_obs_per_image": r"Mean observations per image:\s*([0-9.]+)",
        "mean_error": r"Mean reprojection error:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            found[key] = float(match.group(1))

    assert found, f"model_analyzer output did not match any known field:\n{output}"

    if "images" in found:
        assert found["images"] == model.num_registered_images
    if "points" in found:
        assert found["points"] == model.num_points3d
    if "observations" in found:
        assert found["observations"] == model.num_observations
    if "mean_track_length" in found:
        assert found["mean_track_length"] == pytest.approx(model.mean_track_length(), abs=0.01)
    if "mean_obs_per_image" in found:
        assert found["mean_obs_per_image"] == pytest.approx(
            model.mean_observations_per_image(), abs=0.01
        )
    if "mean_error" in found:
        assert found["mean_error"] == pytest.approx(model.mean_reprojection_error(), abs=0.01)
