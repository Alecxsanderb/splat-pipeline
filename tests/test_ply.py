import numpy as np
import pytest

from pipeline.ply import PlyError, concatenate, read_ply, write_ply

from .splat_fixtures import gaussian_property_names, write_gaussian_ply


def test_reads_every_property_of_a_3dgs_ply(tmp_path):
    xyz = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    fields = write_gaussian_ply(tmp_path / "a.ply", xyz)

    cloud = read_ply(tmp_path / "a.ply")

    assert len(cloud) == 3
    # 3 position + 3 normal + 3 DC + 45 rest + opacity + 3 scale + 4 rot = 62
    assert cloud.property_names == gaussian_property_names()
    assert len(cloud.properties) == 62
    for name, expected in fields.items():
        np.testing.assert_allclose(cloud.data[name], expected, rtol=1e-6)
    np.testing.assert_allclose(cloud.xyz, xyz)


def test_round_trip_preserves_all_properties_and_order(tmp_path):
    xyz = np.random.default_rng(0).random((25, 3)) * 10
    fields = write_gaussian_ply(tmp_path / "a.ply", xyz, seed=3)

    original = read_ply(tmp_path / "a.ply")
    write_ply(tmp_path / "b.ply", original)
    reloaded = read_ply(tmp_path / "b.ply")

    assert reloaded.properties == original.properties
    assert len(reloaded) == len(original)
    for name in fields:
        np.testing.assert_allclose(reloaded.data[name], fields[name], rtol=1e-6)


@pytest.mark.parametrize("sh_rest", [0, 9, 24, 45])
def test_handles_any_spherical_harmonic_degree(tmp_path, sh_rest):
    xyz = np.zeros((4, 3))
    write_gaussian_ply(tmp_path / "a.ply", xyz, sh_rest=sh_rest)

    cloud = read_ply(tmp_path / "a.ply")

    rest = [n for n in cloud.property_names if n.startswith("f_rest_")]
    assert len(rest) == sh_rest
    assert len(cloud.properties) == 17 + sh_rest


def test_reads_ascii_ply(tmp_path):
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    write_gaussian_ply(tmp_path / "a.ply", xyz, sh_rest=3, fmt="ascii")

    cloud = read_ply(tmp_path / "a.ply")

    assert len(cloud) == 2
    np.testing.assert_allclose(cloud.xyz, xyz)


def test_reads_big_endian_ply(tmp_path):
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    write_gaussian_ply(tmp_path / "a.ply", xyz, sh_rest=3, fmt="binary_big_endian")

    cloud = read_ply(tmp_path / "a.ply")

    np.testing.assert_allclose(cloud.xyz, xyz)
    # Written back out, it must be little-endian and still correct.
    write_ply(tmp_path / "b.ply", cloud)
    assert b"binary_little_endian" in (tmp_path / "b.ply").read_bytes()[:64]
    np.testing.assert_allclose(read_ply(tmp_path / "b.ply").xyz, xyz)


def test_select_keeps_all_properties(tmp_path):
    xyz = np.arange(30, dtype=np.float64).reshape(10, 3)
    fields = write_gaussian_ply(tmp_path / "a.ply", xyz)
    cloud = read_ply(tmp_path / "a.ply")

    mask = np.zeros(10, dtype=bool)
    mask[[1, 4, 7]] = True
    subset = cloud.select(mask)

    assert len(subset) == 3
    assert subset.properties == cloud.properties
    for name in fields:
        np.testing.assert_allclose(subset.data[name], fields[name][[1, 4, 7]], rtol=1e-6)


def test_concatenate_preserves_counts_and_properties(tmp_path):
    write_gaussian_ply(tmp_path / "a.ply", np.zeros((5, 3)), seed=1)
    write_gaussian_ply(tmp_path / "b.ply", np.ones((7, 3)), seed=2)
    a, b = read_ply(tmp_path / "a.ply"), read_ply(tmp_path / "b.ply")

    merged = concatenate([a, b])

    assert len(merged) == 12
    assert merged.properties == a.properties
    for name in a.property_names:
        np.testing.assert_allclose(merged.data[name][:5], a.data[name], rtol=1e-6)
        np.testing.assert_allclose(merged.data[name][5:], b.data[name], rtol=1e-6)


def test_concatenate_rejects_mismatched_sh_degrees(tmp_path):
    # Merging different SH degrees would silently lose appearance data.
    write_gaussian_ply(tmp_path / "a.ply", np.zeros((3, 3)), sh_rest=45)
    write_gaussian_ply(tmp_path / "b.ply", np.zeros((3, 3)), sh_rest=9)

    with pytest.raises(PlyError, match="spherical-harmonic|different property schema"):
        concatenate([read_ply(tmp_path / "a.ply"), read_ply(tmp_path / "b.ply")])


def test_missing_file_raises(tmp_path):
    with pytest.raises(PlyError, match="not found"):
        read_ply(tmp_path / "nope.ply")


def test_truncated_body_raises(tmp_path):
    write_gaussian_ply(tmp_path / "a.ply", np.zeros((10, 3)))
    data = (tmp_path / "a.ply").read_bytes()
    (tmp_path / "a.ply").write_bytes(data[: len(data) - 200])

    with pytest.raises(PlyError, match="truncated"):
        read_ply(tmp_path / "a.ply")


def test_non_ply_file_raises(tmp_path):
    (tmp_path / "a.ply").write_bytes(b"this is not a ply file\n")

    with pytest.raises(PlyError, match="not a PLY"):
        read_ply(tmp_path / "a.ply")


def test_list_property_is_rejected_rather_than_mangled(tmp_path):
    # A mesh PLY has variable-stride records; guessing would corrupt every row.
    (tmp_path / "a.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"element face 1\nproperty list uchar int vertex_indices\nend_header\n"
    )

    with pytest.raises(PlyError, match="list property"):
        read_ply(tmp_path / "a.ply")


def test_ply_without_positions_is_rejected(tmp_path):
    (tmp_path / "a.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property float red\nend_header\n" + b"\x00" * 4
    )

    with pytest.raises(PlyError, match="no x/y/z|no x"):
        read_ply(tmp_path / "a.ply")
