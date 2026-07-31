import numpy as np
import pytest

from pipeline.colmap_model import (
    INVALID_POINT3D_ID,
    ColmapModelError,
    InconsistentModelError,
    MissingModelFileError,
    TruncatedFileError,
    UnknownCameraModelError,
    find_models,
    qvec_to_rotmat,
    read_cameras_binary,
    read_images_binary,
    read_model,
    read_points3d_binary,
)

from .colmap_fixtures import (
    FixtureCamera,
    FixtureImage,
    FixtureModel,
    FixturePoint3D,
    append_garbage,
    build_fragmented_model,
    build_healthy_model,
    make_consistent,
    set_leading_count,
    truncate_file,
    write_cameras_bin,
    write_images_bin,
    write_model_bin,
    write_points3d_bin,
)

# ---------------------------------------------------------------------------
# Golden bytes: pins the on-disk layout independently of BOTH the reader and
# the fixture writer, so the two cannot silently drift together.
# ---------------------------------------------------------------------------

GOLDEN_CAMERAS_BIN = (
    b"\x01\x00\x00\x00\x00\x00\x00\x00"  # num_cameras = 1
    b"\x07\x00\x00\x00"  # camera_id = 7
    b"\x01\x00\x00\x00"  # model_id = 1 (PINHOLE, signed int32)
    b"\x80\x07\x00\x00\x00\x00\x00\x00"  # width = 1920
    b"\x38\x04\x00\x00\x00\x00\x00\x00"  # height = 1080
    b"\x00\x00\x00\x00\x00\xc0\x92\x40"  # fx = 1200.0
    b"\x00\x00\x00\x00\x00\xc0\x92\x40"  # fy = 1200.0
    b"\x00\x00\x00\x00\x00\x00\x8e\x40"  # cx = 960.0
    b"\x00\x00\x00\x00\x00\xe0\x80\x40"  # cy = 540.0
)

GOLDEN_POINTS3D_BIN = (
    b"\x01\x00\x00\x00\x00\x00\x00\x00"  # num_points = 1
    b"\x2a\x00\x00\x00\x00\x00\x00\x00"  # point3D_id = 42
    b"\x00\x00\x00\x00\x00\x00\xf0\x3f"  # x = 1.0
    b"\x00\x00\x00\x00\x00\x00\x00\x40"  # y = 2.0
    b"\x00\x00\x00\x00\x00\x00\x08\x40"  # z = 3.0
    b"\x0a\x14\x1e"  # rgb = (10, 20, 30) -- 3 unaligned bytes
    b"\x00\x00\x00\x00\x00\x00\xe0\x3f"  # error = 0.5
    b"\x02\x00\x00\x00\x00\x00\x00\x00"  # track_length = 2
    b"\x01\x00\x00\x00\x00\x00\x00\x00"  # (image 1, point2D 0)
    b"\x02\x00\x00\x00\x01\x00\x00\x00"  # (image 2, point2D 1)
)


def test_golden_cameras_bytes_round_trip_both_directions(tmp_path):
    path = tmp_path / "cameras.bin"
    write_cameras_bin(path, [FixtureCamera(camera_id=7, model="PINHOLE")])
    assert path.read_bytes() == GOLDEN_CAMERAS_BIN

    path.write_bytes(GOLDEN_CAMERAS_BIN)
    cameras = read_cameras_binary(path)
    assert list(cameras) == [7]
    camera = cameras[7]
    assert camera.model_id == 1
    assert camera.model_name == "PINHOLE"
    assert (camera.width, camera.height) == (1920, 1080)
    assert camera.params == (1200.0, 1200.0, 960.0, 540.0)
    assert camera.focal_xy == (1200.0, 1200.0)
    assert camera.principal_point == (960.0, 540.0)


def test_golden_points3d_bytes_round_trip_both_directions(tmp_path):
    path = tmp_path / "points3D.bin"
    write_points3d_bin(
        path,
        [
            FixturePoint3D(
                point3d_id=42,
                xyz=(1.0, 2.0, 3.0),
                rgb=(10, 20, 30),
                error=0.5,
                track=[(1, 0), (2, 1)],
            )
        ],
    )
    assert path.read_bytes() == GOLDEN_POINTS3D_BIN

    path.write_bytes(GOLDEN_POINTS3D_BIN)
    points = read_points3d_binary(path)
    assert len(points) == 1
    assert points.num_observations == 2
    point = points[42]
    assert tuple(point.xyz) == (1.0, 2.0, 3.0)
    assert tuple(point.rgb) == (10, 20, 30)
    assert point.error == 0.5
    assert point.track.tolist() == [[1, 0], [2, 1]]


def test_points3d_record_header_is_51_bytes_not_padded_to_56():
    # A numpy dtype with align=True would round this to 56 and corrupt every
    # record; this asserts the packed size the reader relies on.
    from pipeline.colmap_model import _POINT3D_HEADER

    assert _POINT3D_HEADER.size == 51


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_healthy_model(tmp_path):
    fixture = build_healthy_model(num_images=6, num_points=40)
    model_dir = write_model_bin(tmp_path / "0", fixture)

    model = read_model(model_dir)

    assert model.num_registered_images == 6
    assert model.num_points3d == len(fixture.points3d)
    assert model.num_observations == sum(len(p.track) for p in fixture.points3d)
    for expected in fixture.images:
        image = model.images[expected.image_id]
        assert image.name == expected.name
        assert image.camera_id == expected.camera_id
        assert image.num_points2d == len(expected.points2d)
        np.testing.assert_allclose(image.qvec, expected.qvec)
        np.testing.assert_allclose(image.tvec, expected.tvec)


@pytest.mark.parametrize("model_name", sorted(
    ["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE",
     "FULL_OPENCV", "FOV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE"]
))
def test_every_camera_model_round_trips_with_correct_param_count(tmp_path, model_name):
    from .colmap_fixtures import CAMERA_MODEL_NUM_PARAMS

    path = tmp_path / "cameras.bin"
    write_cameras_bin(path, [FixtureCamera(camera_id=1, model=model_name)])

    camera = read_cameras_binary(path)[1]

    assert camera.model_name == model_name
    assert len(camera.params) == CAMERA_MODEL_NUM_PARAMS[model_name]


def test_invalid_point3d_sentinel_is_preserved_and_excluded(tmp_path):
    image = FixtureImage(
        image_id=1,
        name="video_test/a.jpg",
        points2d=[(1.0, 2.0, 5), (3.0, 4.0, None), (5.0, 6.0, None)],
    )
    path = tmp_path / "images.bin"
    write_images_bin(path, [image])

    parsed = read_images_binary(path)[1]

    assert parsed.point3d_ids.tolist() == [5, INVALID_POINT3D_ID, INVALID_POINT3D_ID]
    assert parsed.num_valid_points3d == 1
    assert parsed.valid_point3d_ids().tolist() == [5]


@pytest.mark.parametrize(
    "name",
    ["video_4k/clip_000001.jpg", "a photo with spaces.jpg", "café/naïve.jpg", "x.jpg"],
)
def test_image_names_survive_round_trip(tmp_path, name):
    path = tmp_path / "images.bin"
    write_images_bin(path, [FixtureImage(image_id=1, name=name, points2d=[(0.0, 0.0, None)])])

    assert read_images_binary(path)[1].name == name


def test_image_group_is_leading_path_component(tmp_path):
    path = tmp_path / "images.bin"
    write_images_bin(
        path,
        [
            FixtureImage(image_id=1, name="video_4k/clip_000001.jpg", points2d=[]),
            FixtureImage(image_id=2, name="loose.jpg", points2d=[]),
        ],
    )

    images = read_images_binary(path)

    assert images[1].group == "video_4k"
    assert images[2].group == ""


def test_zero_point2d_image_and_length_one_track(tmp_path):
    fixture = FixtureModel(
        cameras=[FixtureCamera(camera_id=1)],
        images=[
            FixtureImage(image_id=1, name="g/a.jpg", points2d=[]),
            FixtureImage(image_id=2, name="g/b.jpg", points2d=[(0.0, 0.0, 1)]),
        ],
        points3d=[FixturePoint3D(point3d_id=1, xyz=(0.0, 0.0, 0.0), track=[(2, 0)])],
    )
    model_dir = write_model_bin(tmp_path / "0", fixture)

    model = read_model(model_dir)

    assert model.images[1].num_points2d == 0
    assert model.points3d.track_lengths().tolist() == [1]


def test_read_points2d_false_keeps_counts_but_drops_arrays(tmp_path):
    fixture = build_healthy_model(num_images=4, num_points=20)
    model_dir = write_model_bin(tmp_path / "0", fixture)

    full = read_images_binary(model_dir / "images.bin")
    lean = read_images_binary(model_dir / "images.bin", read_points2d=False)

    assert {i: img.num_points2d for i, img in lean.items()} == {
        i: img.num_points2d for i, img in full.items()
    }
    assert all(img.points2d is None for img in lean.values())
    with pytest.raises(ValueError, match="not read"):
        _ = lean[1].xys


def test_camera_center_matches_manual_computation(tmp_path):
    qvec = (0.5, 0.5, 0.5, 0.5)
    tvec = (1.0, 2.0, 3.0)
    path = tmp_path / "images.bin"
    write_images_bin(
        path, [FixtureImage(image_id=1, name="g/a.jpg", qvec=qvec, tvec=tvec, points2d=[])]
    )

    center = read_images_binary(path)[1].projection_center()

    expected = -qvec_to_rotmat(qvec).T @ np.array(tvec)
    np.testing.assert_allclose(center, expected)


def test_qvec_to_rotmat_is_orthonormal():
    rot = qvec_to_rotmat((0.5, 0.5, 0.5, 0.5))

    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rot), 1.0)


def test_non_unit_quaternion_is_normalized_with_warning(caplog):
    with caplog.at_level("WARNING"):
        rot = qvec_to_rotmat((2.0, 0.0, 0.0, 0.0))

    np.testing.assert_allclose(rot, np.eye(3), atol=1e-12)
    assert any("norm" in r.message for r in caplog.records)


def test_zero_quaternion_raises():
    with pytest.raises(InconsistentModelError):
        qvec_to_rotmat((0.0, 0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Error paths that real COLMAP cannot produce for us
# ---------------------------------------------------------------------------


def test_missing_file_raises(tmp_path):
    with pytest.raises(MissingModelFileError):
        read_cameras_binary(tmp_path / "cameras.bin")


@pytest.mark.parametrize("keep", [4, 12, 30, 40])
def test_truncated_cameras_raises_with_offset(tmp_path, keep):
    path = tmp_path / "cameras.bin"
    write_cameras_bin(path, [FixtureCamera(camera_id=1, model="OPENCV")])
    truncate_file(path, keep)

    with pytest.raises(TruncatedFileError, match="truncated|declares"):
        read_cameras_binary(path)


def test_truncated_image_name_raises(tmp_path):
    path = tmp_path / "images.bin"
    write_images_bin(path, [FixtureImage(image_id=1, name="video/a.jpg", points2d=[])])
    truncate_file(path, 8 + 64 + 4)  # cut partway through the name

    with pytest.raises(TruncatedFileError):
        read_images_binary(path)


def test_truncated_point2d_block_raises(tmp_path):
    path = tmp_path / "images.bin"
    write_images_bin(
        path,
        [FixtureImage(image_id=1, name="a.jpg", points2d=[(1.0, 2.0, None)] * 4)],
    )
    truncate_file(path, path.stat().st_size - 30)

    with pytest.raises(TruncatedFileError):
        read_images_binary(path)


def test_inflated_leading_count_fails_fast_without_huge_allocation(tmp_path):
    path = tmp_path / "points3D.bin"
    write_points3d_bin(path, [FixturePoint3D(point3d_id=1, xyz=(0.0, 0.0, 0.0), track=[(1, 0)])])
    set_leading_count(path, 2**40)

    with pytest.raises(TruncatedFileError, match="declares"):
        read_points3d_binary(path)


@pytest.mark.parametrize("writer,name", [
    (lambda p: write_cameras_bin(p, [FixtureCamera(camera_id=1)]), "cameras.bin"),
    (lambda p: write_images_bin(p, [FixtureImage(image_id=1, name="a.jpg")]), "images.bin"),
    (lambda p: write_points3d_bin(p, [FixturePoint3D(1, (0.0, 0.0, 0.0))]), "points3D.bin"),
])
def test_trailing_garbage_is_rejected(tmp_path, writer, name):
    path = tmp_path / name
    writer(path)
    append_garbage(path)

    readers = {
        "cameras.bin": read_cameras_binary,
        "images.bin": read_images_binary,
        "points3D.bin": read_points3d_binary,
    }
    with pytest.raises(ColmapModelError, match="trailing"):
        readers[name](path)


@pytest.mark.parametrize("bad_id", [99, -1])
def test_unknown_camera_model_id_raises(tmp_path, bad_id):
    path = tmp_path / "cameras.bin"
    write_cameras_bin(path, [FixtureCamera(camera_id=1, model=bad_id, params=(1.0, 2.0, 3.0))])

    with pytest.raises(UnknownCameraModelError, match=str(bad_id)):
        read_cameras_binary(path)


def test_duplicate_image_id_raises(tmp_path):
    path = tmp_path / "images.bin"
    write_images_bin(
        path,
        [
            FixtureImage(image_id=1, name="a.jpg", points2d=[]),
            FixtureImage(image_id=1, name="b.jpg", points2d=[]),
        ],
    )

    with pytest.raises(InconsistentModelError, match="duplicate image id"):
        read_images_binary(path)


def test_track_referencing_missing_image_raises(tmp_path):
    fixture = FixtureModel(
        cameras=[FixtureCamera(camera_id=1)],
        images=[FixtureImage(image_id=1, name="g/a.jpg", points2d=[(0.0, 0.0, 1)])],
        points3d=[FixturePoint3D(point3d_id=1, xyz=(0.0, 0.0, 0.0), track=[(1, 0), (99, 0)])],
    )
    model_dir = write_model_bin(tmp_path / "0", fixture)

    with pytest.raises(InconsistentModelError, match="absent from images.bin"):
        read_model(model_dir)


def test_track_index_out_of_range_raises(tmp_path):
    fixture = FixtureModel(
        cameras=[FixtureCamera(camera_id=1)],
        images=[FixtureImage(image_id=1, name="g/a.jpg", points2d=[(0.0, 0.0, 1)])],
        points3d=[FixturePoint3D(point3d_id=1, xyz=(0.0, 0.0, 0.0), track=[(1, 5)])],
    )
    model_dir = write_model_bin(tmp_path / "0", fixture)

    with pytest.raises(InconsistentModelError, match="index past the end"):
        read_model(model_dir)


def test_bidirectional_mismatch_between_images_and_points3d_raises(tmp_path):
    # Deliberately skips make_consistent: the image claims no 3D point at the
    # index the track points at. This is the check that cross-validates the two
    # parsers against each other.
    fixture = FixtureModel(
        cameras=[FixtureCamera(camera_id=1)],
        images=[
            FixtureImage(image_id=1, name="g/a.jpg", points2d=[(0.0, 0.0, None)]),
            FixtureImage(image_id=2, name="g/b.jpg", points2d=[(0.0, 0.0, 1)]),
        ],
        points3d=[FixturePoint3D(point3d_id=1, xyz=(0.0, 0.0, 0.0), track=[(1, 0), (2, 0)])],
    )
    model_dir = write_model_bin(tmp_path / "0", fixture)

    with pytest.raises(InconsistentModelError, match="disagree"):
        read_model(model_dir)


def test_unknown_camera_reference_raises(tmp_path):
    fixture = FixtureModel(
        cameras=[FixtureCamera(camera_id=1)],
        images=[FixtureImage(image_id=1, name="g/a.jpg", camera_id=42, points2d=[])],
        points3d=[],
    )
    model_dir = write_model_bin(tmp_path / "0", fixture)

    with pytest.raises(InconsistentModelError, match="camera absent"):
        read_model(model_dir)


def test_duplicate_image_names_are_detected(tmp_path):
    fixture = make_consistent(
        FixtureModel(
            cameras=[FixtureCamera(camera_id=1)],
            images=[
                FixtureImage(image_id=1, name="g/same.jpg", points2d=[]),
                FixtureImage(image_id=2, name="g/same.jpg", points2d=[]),
            ],
            points3d=[],
        )
    )
    model = read_model(write_model_bin(tmp_path / "0", fixture))

    with pytest.raises(InconsistentModelError, match="duplicate image name"):
        model.images_by_name()


# ---------------------------------------------------------------------------
# find_models
# ---------------------------------------------------------------------------


def test_find_models_bare_layout(tmp_path):
    write_model_bin(tmp_path / "sparse", build_healthy_model(num_images=3, num_points=10))

    assert find_models(tmp_path / "sparse") == [tmp_path / "sparse"]


def test_find_models_numbered_submodels_sort_numerically(tmp_path):
    sparse = tmp_path / "sparse"
    for i in [0, 1, 2, 9, 10]:
        write_model_bin(sparse / str(i), build_healthy_model(num_images=3, num_points=10))

    assert [p.name for p in find_models(sparse)] == ["0", "1", "2", "9", "10"]


def test_find_models_raises_on_partially_written_model(tmp_path):
    sparse = tmp_path / "sparse"
    write_model_bin(sparse / "0", build_healthy_model(num_images=3, num_points=10))
    write_model_bin(sparse / "1", build_healthy_model(num_images=3, num_points=10))
    (sparse / "1" / "points3D.bin").unlink()

    with pytest.raises(MissingModelFileError, match="partially written"):
        find_models(sparse)


def test_find_models_ignores_unrelated_directories(tmp_path):
    sparse = tmp_path / "sparse"
    write_model_bin(sparse / "0", build_healthy_model(num_images=3, num_points=10))
    (sparse / "dense").mkdir()
    (sparse / "project.ini").write_text("")

    assert [p.name for p in find_models(sparse)] == ["0"]


def test_find_models_suggests_conversion_for_text_models(tmp_path):
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    (sparse / "cameras.txt").write_text("")

    with pytest.raises(MissingModelFileError, match="model_converter"):
        find_models(sparse)


def test_find_models_missing_dir_raises(tmp_path):
    with pytest.raises(MissingModelFileError):
        find_models(tmp_path / "nope")


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def test_model_aggregates(tmp_path):
    fixture = build_healthy_model(num_images=6, num_points=40, error=0.5)
    model = read_model(write_model_bin(tmp_path / "0", fixture))

    total_obs = sum(len(p.track) for p in fixture.points3d)
    assert model.num_observations == total_obs
    assert model.mean_track_length() == pytest.approx(total_obs / len(fixture.points3d))
    assert model.mean_observations_per_image() == pytest.approx(total_obs / 6)
    assert model.mean_reprojection_error() == pytest.approx(0.5)

    per_image = model.observations_per_image()
    assert set(per_image) == set(model.images)
    assert sum(per_image.values()) == total_obs


def test_camera_centers_are_ordered_by_image_id(tmp_path):
    fixture = build_fragmented_model(component_sizes=(4, 3))
    model = read_model(write_model_bin(tmp_path / "0", fixture))

    image_ids, centers = model.camera_centers()

    assert image_ids == sorted(model.images)
    assert centers.shape == (len(model.images), 3)
    assert np.all(np.isfinite(centers))
