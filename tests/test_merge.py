import argparse
import json

import numpy as np
import pytest

from pipeline.boxes import BoundingBox
from pipeline.commands import merge
from pipeline.config import MergeConfig, Paths, PipelineConfig
from pipeline.ply import read_ply

from .splat_fixtures import gaussian_property_names, write_gaussian_ply


def _config(tmp_path, **merge_kwargs):
    return PipelineConfig(
        paths=Paths(
            input_video_dir=tmp_path / "input" / "video",
            input_photo_dir=tmp_path / "input" / "photos",
            workdir=tmp_path / "work",
            output_dir=tmp_path / "output",
        ),
        merge=MergeConfig(**{"crop_margin": 0.0, **merge_kwargs}),
    )


def _args(**overrides):
    base = {
        "chunks_dir": None, "output": None, "ply_name": None,
        "crop_margin": None, "no_crop": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _make_chunk(chunks_dir, name, box, xyz, *, ply_name="point_cloud.ply", nested=False, seed=0):
    """Write a chunk directory holding a manifest and a trained splat."""
    chunk_dir = chunks_dir / name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "name": name,
        "box": box.to_dict(),
        "num_images": 10,
        "num_points": len(xyz),
    }) + "\n")
    target = chunk_dir / ("point_cloud/iteration_30000" if nested else "") / ply_name
    fields = write_gaussian_ply(target, np.asarray(xyz, dtype=np.float64), seed=seed)
    return chunk_dir, fields


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def test_crop_removes_gaussians_outside_the_box(tmp_path):
    box = BoundingBox("a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    xyz = np.array([
        [5.0, 5.0, 5.0],     # inside
        [0.0, 0.0, 0.0],     # on the boundary
        [11.0, 5.0, 5.0],    # outside +x
        [-1.0, 5.0, 5.0],    # outside -x
        [5.0, 5.0, 50.0],    # far outside
    ])
    write_gaussian_ply(tmp_path / "a.ply", xyz)

    cropped = merge.crop_to_box(read_ply(tmp_path / "a.ply"), box, margin=0.0)

    assert len(cropped) == 2
    np.testing.assert_allclose(cropped.xyz, xyz[:2])


def test_crop_margin_widens_the_kept_region(tmp_path):
    box = BoundingBox("a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    xyz = np.array([[5.0, 5.0, 5.0], [10.5, 5.0, 5.0], [12.0, 5.0, 5.0]])
    write_gaussian_ply(tmp_path / "a.ply", xyz)
    cloud = read_ply(tmp_path / "a.ply")

    assert len(merge.crop_to_box(cloud, box, margin=0.0)) == 1
    assert len(merge.crop_to_box(cloud, box, margin=1.0)) == 2
    assert len(merge.crop_to_box(cloud, box, margin=5.0)) == 3


def test_crop_preserves_every_property(tmp_path):
    box = BoundingBox("a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    xyz = np.array([[1.0, 1.0, 1.0], [99.0, 99.0, 99.0], [2.0, 2.0, 2.0]])
    fields = write_gaussian_ply(tmp_path / "a.ply", xyz)

    cropped = merge.crop_to_box(read_ply(tmp_path / "a.ply"), box, margin=0.0)

    assert cropped.property_names == gaussian_property_names()
    for name in fields:
        np.testing.assert_allclose(cropped.data[name], fields[name][[0, 2]], rtol=1e-6)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def test_merge_concatenates_cropped_chunks(tmp_path):
    config = _config(tmp_path)
    chunks = config.paths.output_dir / "chunks"
    box_a = BoundingBox("room_a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    box_b = BoundingBox("room_b", (20.0, 0.0, 0.0), (30.0, 10.0, 10.0))
    # Each chunk was trained with overlap, so it holds Gaussians belonging to
    # its neighbour; those must be cropped away rather than duplicated.
    _make_chunk(chunks, "room_a", box_a, [[1, 1, 1], [2, 2, 2], [25.0, 5, 5]], seed=1)
    _make_chunk(chunks, "room_b", box_b, [[21, 1, 1], [22, 2, 2], [5.0, 5, 5]], seed=2)

    exit_code = merge.run(_args(), config)

    assert exit_code == 0
    merged = read_ply(config.paths.output_dir / "merged.ply")
    assert len(merged) == 4
    xs = sorted(merged.xyz[:, 0].tolist())
    assert xs == pytest.approx([1.0, 2.0, 21.0, 22.0])


def test_merge_reports_counts_before_and_after(tmp_path, caplog):
    config = _config(tmp_path)
    chunks = config.paths.output_dir / "chunks"
    box = BoundingBox("room_a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    _make_chunk(chunks, "room_a", box, [[1, 1, 1], [2, 2, 2], [99.0, 99, 99]])

    with caplog.at_level("INFO"):
        merge.run(_args(), config)

    messages = " ".join(r.message for r in caplog.records)
    assert "3 Gaussians in, 2 out" in messages
    assert "3 ->         2 Gaussians (1 cropped away" in messages


def test_merge_preserves_every_property_field(tmp_path):
    config = _config(tmp_path)
    chunks = config.paths.output_dir / "chunks"
    box_a = BoundingBox("room_a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    box_b = BoundingBox("room_b", (20.0, 0.0, 0.0), (30.0, 10.0, 10.0))
    _, fields_a = _make_chunk(chunks, "room_a", box_a, [[1, 1, 1], [2, 2, 2]], seed=1)
    _, fields_b = _make_chunk(chunks, "room_b", box_b, [[21, 1, 1], [22, 2, 2]], seed=2)

    merge.run(_args(), config)
    merged = read_ply(config.paths.output_dir / "merged.ply")

    assert merged.property_names == gaussian_property_names()
    assert len(merged.properties) == 62
    assert len(merged) == 4
    # Nothing was cropped here, so every value from both chunks must appear.
    for name in gaussian_property_names():
        np.testing.assert_allclose(merged.data[name][:2], fields_a[name], rtol=1e-6)
        np.testing.assert_allclose(merged.data[name][2:], fields_b[name], rtol=1e-6)


def test_merge_no_crop_keeps_the_overlap(tmp_path):
    config = _config(tmp_path)
    chunks = config.paths.output_dir / "chunks"
    box = BoundingBox("room_a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    _make_chunk(chunks, "room_a", box, [[1, 1, 1], [99.0, 99, 99]])

    merge.run(_args(no_crop=True), config)

    assert len(read_ply(config.paths.output_dir / "merged.ply")) == 2


def test_merge_uses_the_configured_crop_margin(tmp_path):
    config = _config(tmp_path, crop_margin=0.0)
    chunks = config.paths.output_dir / "chunks"
    box = BoundingBox("room_a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    _make_chunk(chunks, "room_a", box, [[1, 1, 1], [10.5, 5, 5]])

    merge.run(_args(crop_margin=1.0), config)

    assert len(read_ply(config.paths.output_dir / "merged.ply")) == 2


def test_merge_finds_nested_3dgs_output(tmp_path):
    # Real 3DGS writes point_cloud/iteration_30000/point_cloud.ply.
    config = _config(tmp_path)
    chunks = config.paths.output_dir / "chunks"
    box = BoundingBox("room_a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    _make_chunk(chunks, "room_a", box, [[1, 1, 1], [2, 2, 2]], nested=True)

    assert merge.run(_args(), config) == 0
    assert len(read_ply(config.paths.output_dir / "merged.ply")) == 2


def test_merge_skips_untrained_chunks_with_a_warning(tmp_path, caplog):
    config = _config(tmp_path)
    chunks = config.paths.output_dir / "chunks"
    box_a = BoundingBox("room_a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    _make_chunk(chunks, "room_a", box_a, [[1, 1, 1]])
    # room_b has a manifest but was never trained.
    (chunks / "room_b").mkdir(parents=True)
    (chunks / "room_b" / "manifest.json").write_text(json.dumps({
        "name": "room_b",
        "box": BoundingBox("room_b", (20.0, 0.0, 0.0), (30.0, 10.0, 10.0)).to_dict(),
    }))

    with caplog.at_level("WARNING"):
        exit_code = merge.run(_args(), config)

    assert exit_code == 0
    assert any("has this chunk been trained" in r.message for r in caplog.records)
    assert len(read_ply(config.paths.output_dir / "merged.ply")) == 1


def test_merge_with_no_chunks_is_nothing_to_do(tmp_path):
    config = _config(tmp_path)

    assert merge.run(_args(), config) == 3


def test_merge_warns_when_cropping_removes_everything(tmp_path, caplog):
    # The usual cause is a splat that is not in the same coordinate frame as
    # the model it was chunked from.
    config = _config(tmp_path)
    chunks = config.paths.output_dir / "chunks"
    box = BoundingBox("room_a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    _make_chunk(chunks, "room_a", box, [[500.0, 500.0, 500.0]])

    with caplog.at_level("WARNING"):
        merge.run(_args(), config)

    assert any("removed every Gaussian" in r.message for r in caplog.records)


def test_merge_rejects_chunks_with_different_sh_degrees(tmp_path, caplog):
    config = _config(tmp_path)
    chunks = config.paths.output_dir / "chunks"
    box_a = BoundingBox("room_a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    box_b = BoundingBox("room_b", (20.0, 0.0, 0.0), (30.0, 10.0, 10.0))
    _make_chunk(chunks, "room_a", box_a, [[1, 1, 1]])
    chunk_b = chunks / "room_b"
    chunk_b.mkdir(parents=True)
    (chunk_b / "manifest.json").write_text(json.dumps({"name": "room_b", "box": box_b.to_dict()}))
    write_gaussian_ply(chunk_b / "point_cloud.ply", np.array([[21.0, 1.0, 1.0]]), sh_rest=9)

    with caplog.at_level("ERROR"):
        exit_code = merge.run(_args(), config)

    assert exit_code == 1
    assert any("spherical-harmonic" in r.message for r in caplog.records)


def test_merge_output_path_is_configurable(tmp_path):
    config = _config(tmp_path)
    chunks = config.paths.output_dir / "chunks"
    box = BoundingBox("room_a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    _make_chunk(chunks, "room_a", box, [[1, 1, 1]])

    merge.run(_args(output=tmp_path / "custom" / "out.ply"), config)

    assert (tmp_path / "custom" / "out.ply").is_file()


def test_merged_ply_is_readable_by_a_fresh_reader(tmp_path):
    config = _config(tmp_path)
    chunks = config.paths.output_dir / "chunks"
    box = BoundingBox("room_a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    _make_chunk(chunks, "room_a", box, [[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    merge.run(_args(), config)
    merged = read_ply(config.paths.output_dir / "merged.ply")

    assert len(merged) == 3
    assert merged.data.dtype.names == tuple(gaussian_property_names())
    assert np.all(np.isfinite(merged.xyz))
