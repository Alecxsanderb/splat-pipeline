import argparse
import csv
import json
import sqlite3

import numpy as np
import pytest

from pipeline.commands import verify
from pipeline.config import Paths, PipelineConfig, VerifyConfig

from .colmap_fixtures import (
    FixtureCamera,
    FixtureImage,
    FixtureModel,
    FixturePoint3D,
    build_fragmented_model,
    build_healthy_model,
    make_consistent,
    write_model_bin,
)


def _config(tmp_path, **verify_kwargs):
    return PipelineConfig(
        paths=Paths(
            input_video_dir=tmp_path / "input" / "video",
            input_photo_dir=tmp_path / "input" / "photos",
            workdir=tmp_path / "work",
            output_dir=tmp_path / "output",
        ),
        verify=VerifyConfig(**{"min_common_points": 3, "min_component_size": 2, **verify_kwargs}),
    )


def _place_input_images(config, model, extra_unregistered=0, group="video_test"):
    """Create the on-disk images that COLMAP would have been pointed at."""
    images_root = config.paths.output_dir / "images"
    for image in model.images:
        path = images_root / image.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake jpeg")
    for i in range(extra_unregistered):
        path = images_root / group / f"unregistered_{i:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake jpeg")
    return images_root


def _args(**overrides):
    base = {
        "sparse_dir": None, "images_dir": None, "model": None,
        "min_registration": None, "min_common_points": None, "no_plot": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _report(config):
    return json.loads(
        (config.paths.workdir / "verify" / "verify_report.json").read_text()
    )


def _check(report, name):
    return next(c for c in report["checks"] if c["name"] == name)


# ---------------------------------------------------------------------------
# Diagnosis 1: a healthy reconstruction
# ---------------------------------------------------------------------------


def test_healthy_model_passes(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=12, num_points=120, track_length=5)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    exit_code = verify.run(_args(), config)

    assert exit_code == 0
    report = _report(config)
    assert report["status"] == "pass"
    assert report["primary_model"]["registered_images"] == 12
    assert report["primary_model"]["registration_rate"] == 1.0
    assert report["primary_model"]["components"]["counted_for_pass_fail"] == 1
    assert _check(report, "connectivity")["passed"] is True
    assert _check(report, "registration_rate")["passed"] is True
    assert _check(report, "single_model")["passed"] is True


def test_healthy_model_writes_plot_and_reports(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=12, num_points=120)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    verify.run(_args(), config)

    report_dir = config.paths.workdir / "verify"
    png = report_dir / "camera_positions_top_down.png"
    assert png.is_file()
    assert png.stat().st_size > 5000  # a real figure, not an empty canvas
    assert (report_dir / "registration_by_group.csv").is_file()
    assert _report(config)["primary_model"]["plot"]["status"] == "written"


# ---------------------------------------------------------------------------
# Diagnosis 2: the scene fragmented
# ---------------------------------------------------------------------------


def test_fragmented_model_fails_and_reports_two_components(tmp_path):
    config = _config(tmp_path)
    fixture = build_fragmented_model(component_sizes=(8, 6), points_per_component=60)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    exit_code = verify.run(_args(), config)

    assert exit_code == 1
    report = _report(config)
    assert report["status"] == "fail"
    components = report["primary_model"]["components"]
    assert components["count"] == 2
    assert components["counted_for_pass_fail"] == 2
    assert sorted(components["sizes"], reverse=True) == [8, 6]
    assert _check(report, "connectivity")["passed"] is False


def test_fragmented_model_writes_cross_component_candidate_pairs(tmp_path):
    config = _config(tmp_path)
    fixture = build_fragmented_model(component_sizes=(8, 6), points_per_component=60)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    verify.run(_args(), config)

    pair_file = config.paths.workdir / "verify" / "candidate_pairs.txt"
    assert pair_file.is_file()
    lines = [line for line in pair_file.read_text().splitlines() if line.strip()]
    assert lines

    # The two components use distinct name prefixes, so every proposed pair
    # must straddle them to be useful.
    for line in lines:
        left, right = line.split()
        assert left.split("/")[0] != right.split("/")[0], line

    candidates = config.paths.workdir / "verify" / "boundary_candidates.csv"
    rows = list(csv.DictReader(candidates.open()))
    assert rows
    assert all(row["component_a"] != row["component_b"] for row in rows)


def test_weakly_linked_components_are_ranked_by_shared_points(tmp_path):
    config = _config(tmp_path, min_common_points=25)
    fixture = build_fragmented_model(
        component_sizes=(6, 6), points_per_component=60, bridge_tracks=4
    )
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    exit_code = verify.run(_args(), config)

    assert exit_code == 1
    rows = list(
        csv.DictReader((config.paths.workdir / "verify" / "boundary_candidates.csv").open())
    )
    shared = [int(r["shared_points"]) for r in rows if r["source"] == "shared_points"]
    assert shared, "bridge tracks should surface as shared-point boundary pairs"
    assert shared == sorted(shared, reverse=True)


def test_tiny_stray_component_is_reported_but_does_not_fail(tmp_path):
    config = _config(tmp_path, min_common_points=3, min_component_size=4)
    fixture = build_fragmented_model(component_sizes=(10, 2), points_per_component=40)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    exit_code = verify.run(_args(), config)

    report = _report(config)
    components = report["primary_model"]["components"]
    assert components["count"] == 2  # both are reported
    assert components["counted_for_pass_fail"] == 1  # but only the big one counts
    assert _check(report, "connectivity")["passed"] is True
    assert exit_code == 0


# ---------------------------------------------------------------------------
# Diagnosis 3: poor registration
# ---------------------------------------------------------------------------


def test_poor_registration_fails_with_correct_percentage(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=6, num_points=60)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    # 6 registered out of 30 on disk = 20%.
    _place_input_images(config, fixture, extra_unregistered=24)

    exit_code = verify.run(_args(), config)

    assert exit_code == 1
    report = _report(config)
    assert report["inventory"]["total_images_on_disk"] == 30
    assert report["primary_model"]["registered_images"] == 6
    assert report["primary_model"]["registration_rate"] == pytest.approx(0.2)
    assert _check(report, "registration_rate")["passed"] is False
    assert "20.0%" in _check(report, "registration_rate")["detail"]


def test_registration_threshold_is_configurable(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=8, num_points=80)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture, extra_unregistered=2)  # 80%

    assert verify.run(_args(), config) == 1
    assert verify.run(_args(min_registration=0.75), config) == 0


def test_per_group_registration_rates_identify_the_failing_room(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(
        num_images=8, num_points=80, name_template="video_kitchen/frame_{index:06d}.jpg"
    )
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)
    # A whole second room that registered nothing at all.
    for i in range(5):
        path = config.paths.output_dir / "images" / "video_garage" / f"frame_{i:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake jpeg")

    exit_code = verify.run(_args(min_registration=0.5), config)

    assert exit_code == 1
    rows = {
        r["group"]: r
        for r in csv.DictReader(
            (config.paths.workdir / "verify" / "registration_by_group.csv").open()
        )
    }
    assert rows["video_kitchen"]["registered"] == "8"
    assert rows["video_garage"]["registered"] == "0"
    assert float(rows["video_garage"]["rate"]) == 0.0
    assert _check(_report(config), "group_coverage")["passed"] is False


# ---------------------------------------------------------------------------
# Multiple models, and unverifiable inputs
# ---------------------------------------------------------------------------


def test_multiple_models_is_a_hard_failure(tmp_path):
    config = _config(tmp_path)
    sparse = config.paths.output_dir / "sparse"
    big = build_healthy_model(num_images=8, num_points=80, name_template="a/f_{index:06d}.jpg")
    small = build_healthy_model(num_images=3, num_points=30, name_template="b/f_{index:06d}.jpg")
    write_model_bin(sparse / "0", big)
    write_model_bin(sparse / "1", small)
    _place_input_images(config, big)
    _place_input_images(config, small)

    exit_code = verify.run(_args(), config)

    assert exit_code == 1
    report = _report(config)
    assert len(report["models"]) == 2
    assert _check(report, "single_model")["passed"] is False
    # The largest model is the one analysed in detail.
    assert report["primary_model"]["registered_images"] == 8
    assert (config.paths.workdir / "verify" / "model_1_images.txt").is_file()


def test_missing_sparse_dir_is_unverifiable(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=4, num_points=40)
    _place_input_images(config, fixture)

    exit_code = verify.run(_args(), config)

    assert exit_code == 3
    assert _report(config)["status"] == "unverifiable"


def test_missing_images_dir_is_unverifiable(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=4, num_points=40)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)

    exit_code = verify.run(_args(), config)

    assert exit_code == 3
    report = _report(config)
    assert report["status"] == "unverifiable"
    assert "organize" in report["reason"]


def test_empty_sparse_dir_is_unverifiable(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=4, num_points=40)
    _place_input_images(config, fixture)
    (config.paths.output_dir / "sparse").mkdir(parents=True)

    assert verify.run(_args(), config) == 3


# ---------------------------------------------------------------------------
# Reprojection error handling
# ---------------------------------------------------------------------------


def test_unpopulated_reprojection_error_is_reported_as_unavailable(tmp_path):
    # GLOMAP can leave this field at zero; reporting "mean 0.00 px" would read
    # as a flawless reconstruction, so it must be called out instead.
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=8, num_points=80, error=0.0)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    verify.run(_args(), config)

    errors = _report(config)["primary_model"]["reprojection_error"]
    assert errors["available"] is False
    assert errors["mean"] == 0.0
    assert "no usable reprojection error" in errors["reason"]
    assert _check(_report(config), "reprojection_error")["passed"] is False


def test_reprojection_error_statistics(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=8, num_points=80, error=0.5)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    verify.run(_args(), config)

    errors = _report(config)["primary_model"]["reprojection_error"]
    assert errors["available"] is True
    assert errors["mean"] == pytest.approx(0.5)
    assert errors["median"] == pytest.approx(0.5)
    assert errors["observation_weighted_mean"] == pytest.approx(0.5)


def test_high_reprojection_error_warns_but_does_not_fail(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=8, num_points=80, error=9.0)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    exit_code = verify.run(_args(), config)

    assert exit_code == 0
    check = _check(_report(config), "reprojection_error")
    assert check["passed"] is False
    assert check["severity"] == "warn"


# ---------------------------------------------------------------------------
# Database enrichment
# ---------------------------------------------------------------------------


def _make_database(path, names, matched_pairs=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
        connection.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY)")
        ids = {}
        for i, name in enumerate(sorted(names), start=1):
            connection.execute("INSERT INTO images VALUES (?, ?)", (i, name))
            ids[name] = i
        for a, b in matched_pairs:
            lo, hi = sorted((ids[a], ids[b]))
            connection.execute(
                "INSERT INTO two_view_geometries VALUES (?)", (lo * 2147483647 + hi,)
            )
    return ids


def test_database_gap_is_reported(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=8, num_points=80)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture, extra_unregistered=2)
    # Only 8 of the 10 on-disk images ever made it into the database.
    _make_database(
        config.paths.workdir / "sfm" / "database.db", [i.name for i in fixture.images]
    )

    verify.run(_args(min_registration=0.5), config)

    report = _report(config)
    assert report["inventory"]["total_images_on_disk"] == 10
    assert report["inventory"]["images_in_database"] == 8
    assert _check(report, "database_gap")["passed"] is False


def test_already_matched_pairs_are_excluded_from_the_candidate_file(tmp_path):
    config = _config(tmp_path, min_common_points=25)
    fixture = build_fragmented_model(
        component_sizes=(6, 6), points_per_component=60, bridge_tracks=4
    )
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    pair_file = config.paths.workdir / "verify" / "candidate_pairs.txt"

    # First establish which pairs verify would propose without a database.
    verify.run(_args(), config)
    baseline = [line.split() for line in pair_file.read_text().splitlines() if line.strip()]
    assert baseline

    # Now mark the first proposed pair as already matched.
    already = tuple(baseline[0])
    _make_database(
        config.paths.workdir / "sfm" / "database.db",
        [i.name for i in fixture.images],
        matched_pairs=[already],
    )
    verify.run(_args(), config)

    lines = {tuple(line.split()) for line in pair_file.read_text().splitlines() if line.strip()}
    assert already not in lines, "an already-matched pair must not be proposed for re-matching"

    # ...but it is still visible in the CSV as a diagnostic.
    rows = list(
        csv.DictReader((config.paths.workdir / "verify" / "boundary_candidates.csv").open())
    )
    matched_rows = [r for r in rows if r["already_matched"] == "True"]
    assert matched_rows
    assert {(r["image_a"], r["image_b"]) for r in matched_rows} == {already}


# ---------------------------------------------------------------------------
# Plot degradation
# ---------------------------------------------------------------------------


def test_missing_matplotlib_does_not_change_the_exit_code(tmp_path, monkeypatch):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=8, num_points=80)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib for you")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    exit_code = verify.run(_args(), config)

    assert exit_code == 0
    plot = _report(config)["primary_model"]["plot"]
    assert plot["status"] == "skipped"
    assert plot["reason"] == "matplotlib_missing"


def test_no_plot_flag_skips_rendering(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=8, num_points=80)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    exit_code = verify.run(_args(no_plot=True), config)

    assert exit_code == 0
    assert not (config.paths.workdir / "verify" / "camera_positions_top_down.png").exists()


def test_collinear_cameras_skip_the_plot_without_crashing(tmp_path):
    config = _config(tmp_path)
    images = [
        FixtureImage(
            image_id=i + 1,
            name=f"video_test/frame_{i:06d}.jpg",
            tvec=(float(-i), 0.0, 0.0),
            points2d=[(1.0, 1.0, None), (2.0, 2.0, None)],
        )
        for i in range(4)
    ]
    points = [
        FixturePoint3D(
            point3d_id=p + 1, xyz=(float(p), 0.0, 0.0),
            track=[(i + 1, 0) for i in range(4)],
        )
        for p in range(5)
    ]
    fixture = make_consistent(
        FixtureModel(cameras=[FixtureCamera(camera_id=1)], images=images, points3d=points)
    )
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    exit_code = verify.run(_args(), config)

    assert exit_code == 0
    assert _report(config)["primary_model"]["plot"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# Misc reporting
# ---------------------------------------------------------------------------


def test_stray_images_outside_group_layout_are_flagged(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=8, num_points=80)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)
    (config.paths.output_dir / "images" / "loose.jpg").write_bytes(b"fake jpeg")

    verify.run(_args(min_registration=0.5), config)

    report = _report(config)
    assert report["inventory"]["stray_files"] == ["loose.jpg"]
    assert _check(report, "stray_images")["passed"] is False


def test_extent_and_track_statistics_are_reported(tmp_path):
    config = _config(tmp_path)
    fixture = build_healthy_model(num_images=10, num_points=100, track_length=5)
    write_model_bin(config.paths.output_dir / "sparse" / "0", fixture)
    _place_input_images(config, fixture)

    verify.run(_args(), config)

    primary = _report(config)["primary_model"]
    assert primary["mean_track_length"] > 1.0
    assert len(primary["bounding_box_extent"]) == 3
    assert all(np.isfinite(primary["bounding_box_extent"]))
    assert primary["observations_per_image"]["median"] > 0
    assert primary["components"]["threshold_curve"]


def test_clips_are_qualified_by_group():
    # Two rooms recorded to identically named files both yield clip_000123.jpg;
    # they must not be reported as one capture pass.
    assert verify._clip_of("video_4k/clip_000123.jpg") == "video_4k/clip"
    assert verify._clip_of("video_1080p/clip_000123.jpg") == "video_1080p/clip"
    assert verify._clip_of("photos_48mp/IMG_0042.jpg") == "photos_48mp"
    assert verify._clip_of("loose.jpg") == verify.UNGROUPED


def test_registration_by_clip_separates_identically_named_clips(tmp_path):
    config = _config(tmp_path)
    room_a = build_healthy_model(
        num_images=6, num_points=60, name_template="video_a/clip_{index:06d}.jpg"
    )
    room_b = build_healthy_model(
        num_images=6, num_points=60, name_template="video_b/clip_{index:06d}.jpg",
        first_image_id=100, first_point_id=1000,
    )
    combined = FixtureModel(
        cameras=room_a.cameras,
        images=room_a.images + room_b.images,
        points3d=room_a.points3d + room_b.points3d,
    )
    write_model_bin(config.paths.output_dir / "sparse" / "0", combined)
    _place_input_images(config, combined)

    verify.run(_args(), config)

    clips = {c["key"] for c in _report(config)["primary_model"]["registration_by_clip"]}
    assert clips == {"video_a/clip", "video_b/clip"}


def test_model_flag_selects_a_specific_submodel(tmp_path):
    config = _config(tmp_path)
    sparse = config.paths.output_dir / "sparse"
    big = build_healthy_model(num_images=8, num_points=80, name_template="a/f_{index:06d}.jpg")
    small = build_healthy_model(num_images=3, num_points=30, name_template="b/f_{index:06d}.jpg")
    write_model_bin(sparse / "0", big)
    write_model_bin(sparse / "1", small)
    _place_input_images(config, big)
    _place_input_images(config, small)

    verify.run(_args(model=1), config)

    assert _report(config)["primary_model"]["registered_images"] == 3
