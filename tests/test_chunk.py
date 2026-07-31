import argparse
import json

import numpy as np
import pytest
import yaml

from pipeline.boxes import BoundingBox, BoxError, load_boxes, save_boxes
from pipeline.colmap_model import INVALID_POINT3D_ID, read_model, subset_model, write_model
from pipeline.commands import chunk
from pipeline.config import ChunkConfig, Paths, PipelineConfig

from .colmap_fixtures import build_healthy_model, write_model_bin
from .splat_fixtures import build_two_room_scene


def _config(tmp_path, **chunk_kwargs):
    return PipelineConfig(
        paths=Paths(
            input_video_dir=tmp_path / "input" / "video",
            input_photo_dir=tmp_path / "input" / "photos",
            workdir=tmp_path / "work",
            output_dir=tmp_path / "output",
        ),
        chunk=ChunkConfig(**{"min_points_in_box": 5, **chunk_kwargs}),
    )


def _args(**overrides):
    base = {
        "boxes": None, "sparse_dir": None, "output_dir": None, "model": None,
        "min_points_in_box": None, "suggest": False, "suggest_method": None,
        "clusters": None, "eps": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _scene(tmp_path):
    scene = build_two_room_scene()
    write_model_bin(tmp_path / "output" / "sparse" / "0", scene.model)
    return scene


def _write_boxes(path, rooms):
    save_boxes(path, [BoundingBox(r.name, r.box_min, r.box_max) for r in rooms])
    return path


# ---------------------------------------------------------------------------
# Box definitions
# ---------------------------------------------------------------------------


def test_box_contains_is_inclusive_of_bounds():
    box = BoundingBox("b", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    mask = box.contains(np.array([
        [0.5, 0.5, 0.5],  # inside
        [0.0, 0.0, 0.0],  # on the min corner
        [1.0, 1.0, 1.0],  # on the max corner
        [1.001, 0.5, 0.5],  # just outside
        [-0.001, 0.5, 0.5],  # just outside
    ]))

    assert mask.tolist() == [True, True, True, False, False]


def test_box_expanded_grows_every_side():
    box = BoundingBox("b", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)).expanded(0.5)

    assert box.min == (-0.5, -0.5, -0.5)
    assert box.max == (2.5, 2.5, 2.5)


def test_box_rejects_inverted_bounds():
    with pytest.raises(BoxError, match="below min"):
        BoundingBox("b", (1.0, 0.0, 0.0), (0.0, 1.0, 1.0))


def test_load_boxes_round_trip(tmp_path):
    boxes = [
        BoundingBox("kitchen", (0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
        BoundingBox("hall", (5.0, 0.0, 0.0), (7.0, 2.0, 3.0)),
    ]
    save_boxes(tmp_path / "boxes.yaml", boxes)

    assert load_boxes(tmp_path / "boxes.yaml") == boxes


def test_load_boxes_rejects_duplicate_names(tmp_path):
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"boxes": [
        {"name": "kitchen", "min": [0, 0, 0], "max": [1, 1, 1]},
        {"name": "kitchen", "min": [2, 2, 2], "max": [3, 3, 3]},
    ]}))

    with pytest.raises(BoxError, match="duplicate box name"):
        load_boxes(tmp_path / "b.yaml")


def test_load_boxes_reports_missing_fields(tmp_path):
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"boxes": [{"name": "x", "min": [0, 0, 0]}]}))

    with pytest.raises(BoxError, match="missing max"):
        load_boxes(tmp_path / "b.yaml")


# ---------------------------------------------------------------------------
# Selection -- the through-doorway rule
# ---------------------------------------------------------------------------


def test_selection_includes_cameras_inside_the_box(tmp_path):
    scene = _scene(tmp_path)
    model = read_model(tmp_path / "output" / "sparse" / "0")
    room_a = scene.rooms[0]

    selection = chunk.select_for_box(
        model, BoundingBox(room_a.name, room_a.box_min, room_a.box_max), min_points_in_box=5
    )

    for image_id in room_a.image_ids:
        assert image_id in selection.image_ids


def test_selection_includes_through_doorway_observers(tmp_path):
    # The doorway camera stands in room A, so its centre is outside room B's
    # box entirely -- it can only be selected by the observation rule.
    scene = _scene(tmp_path)
    model = read_model(tmp_path / "output" / "sparse" / "0")
    room_b = scene.rooms[1]
    box = BoundingBox(room_b.name, room_b.box_min, room_b.box_max)

    assert not box.contains(
        model.images[scene.doorway_image_id].projection_center().reshape(1, 3)
    )[0]

    selection = chunk.select_for_box(model, box, min_points_in_box=5)

    assert scene.doorway_image_id in selection.image_ids
    assert selection.through_observers >= 1
    assert selection.observations_in_box[scene.doorway_image_id] == scene.doorway_shared_points


def test_through_doorway_threshold_is_respected(tmp_path):
    scene = _scene(tmp_path)
    model = read_model(tmp_path / "output" / "sparse" / "0")
    room_b = scene.rooms[1]
    box = BoundingBox(room_b.name, room_b.box_min, room_b.box_max)

    # The doorway camera sees exactly `doorway_shared_points` of room B.
    below = chunk.select_for_box(
        model, box, min_points_in_box=scene.doorway_shared_points
    )
    above = chunk.select_for_box(
        model, box, min_points_in_box=scene.doorway_shared_points + 1
    )

    assert scene.doorway_image_id in below.image_ids
    assert scene.doorway_image_id not in above.image_ids


def test_selection_only_takes_points_inside_the_box(tmp_path):
    scene = _scene(tmp_path)
    model = read_model(tmp_path / "output" / "sparse" / "0")
    room_a = scene.rooms[0]

    selection = chunk.select_for_box(
        model, BoundingBox(room_a.name, room_a.box_min, room_a.box_max), min_points_in_box=5
    )

    assert set(selection.point_ids) == set(room_a.point_ids)


# ---------------------------------------------------------------------------
# Subset models are valid and re-readable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("room_index", [0, 1])
def test_subset_model_is_valid_and_re_readable(tmp_path, room_index):
    # room_a is the interesting case: the doorway camera physically stands in
    # it, so it is selected there, yet every point it observes belongs to
    # room_b and is dropped. Those references must be nulled or the chunk is
    # left with dangling point ids.
    scene = _scene(tmp_path)
    model = read_model(tmp_path / "output" / "sparse" / "0")
    room = scene.rooms[room_index]
    selection = chunk.select_for_box(
        model, BoundingBox(room.name, room.box_min, room.box_max), min_points_in_box=5
    )

    submodel, stats = subset_model(
        model, set(selection.image_ids), set(selection.point_ids)
    )
    out = write_model(tmp_path / "sub", submodel)

    # read_model(validate=True) enforces the bidirectional images<->points3D
    # cross-check, so a re-read that succeeds means the subset is consistent.
    reloaded = read_model(out, validate=True)

    assert reloaded.num_registered_images == stats.images
    assert reloaded.num_points3d == stats.points
    assert reloaded.num_observations == stats.observations
    # Every surviving reference must point at a point that is actually present.
    present = set(reloaded.points3d.ids.tolist())
    for image in reloaded.images.values():
        assert set(image.valid_point3d_ids().tolist()) <= present


def test_subset_nulls_references_to_dropped_points(tmp_path):
    scene = _scene(tmp_path)
    model = read_model(tmp_path / "output" / "sparse" / "0")
    room_b_points = set(scene.rooms[1].point_ids)

    # Keep the doorway image but only room A's points; its observations of
    # room B must be reset to the invalid sentinel, not left dangling.
    room_a = scene.rooms[0]
    keep_images = set(room_a.image_ids) | {scene.doorway_image_id}
    submodel, _ = subset_model(model, keep_images, set(room_a.point_ids))

    doorway = submodel.images[scene.doorway_image_id]
    surviving = set(doorway.valid_point3d_ids().tolist())
    assert surviving.isdisjoint(room_b_points)
    assert INVALID_POINT3D_ID in set(doorway.point3d_ids.tolist())


def test_subset_keeps_feature_indices_stable(tmp_path):
    # Track entries address features by index, so the feature table must not be
    # compacted -- otherwise every point2D_idx in the model would shift.
    scene = _scene(tmp_path)
    model = read_model(tmp_path / "output" / "sparse" / "0")
    room_a = scene.rooms[0]
    submodel, _ = subset_model(model, set(room_a.image_ids), set(room_a.point_ids))

    for image_id in room_a.image_ids:
        assert submodel.images[image_id].num_points2d == model.images[image_id].num_points2d
        np.testing.assert_allclose(
            submodel.images[image_id].xys, model.images[image_id].xys
        )


def test_subset_drops_points_below_min_track_length(tmp_path):
    scene = _scene(tmp_path)
    model = read_model(tmp_path / "output" / "sparse" / "0")
    room_a = scene.rooms[0]

    # Only one camera kept: every point now has a track of length 1.
    single = {room_a.image_ids[0]}
    strict, strict_stats = subset_model(
        model, single, set(room_a.point_ids), min_track_length=2
    )
    loose, loose_stats = subset_model(
        model, single, set(room_a.point_ids), min_track_length=1
    )

    assert strict_stats.points == 0
    assert strict_stats.points_dropped_short_track > 0
    assert loose_stats.points > 0
    assert len(strict.points3d) < len(loose.points3d)


def test_subset_only_keeps_referenced_cameras(tmp_path):
    scene = _scene(tmp_path)
    model = read_model(tmp_path / "output" / "sparse" / "0")
    room_a = scene.rooms[0]

    submodel, stats = subset_model(model, set(room_a.image_ids), set(room_a.point_ids))

    assert stats.cameras == 1
    assert set(submodel.cameras) == {img.camera_id for img in submodel.images.values()}


def test_subset_of_no_images_raises(tmp_path):
    scene = _scene(tmp_path)
    model = read_model(tmp_path / "output" / "sparse" / "0")

    with pytest.raises(Exception, match="no images"):
        subset_model(model, set(), set(scene.rooms[0].point_ids))


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def test_chunk_writes_a_model_and_manifest_per_box(tmp_path):
    config = _config(tmp_path)
    scene = _scene(tmp_path)
    boxes_path = _write_boxes(tmp_path / "boxes.yaml", scene.rooms)

    exit_code = chunk.run(_args(boxes=boxes_path), config)

    assert exit_code == 0
    chunks_dir = config.paths.output_dir / "chunks"
    for room in scene.rooms:
        model_dir = chunks_dir / room.name / "sparse" / "0"
        assert read_model(model_dir).num_registered_images > 0

        manifest = json.loads((chunks_dir / room.name / "manifest.json").read_text())
        assert manifest["name"] == room.name
        assert manifest["box"]["min"] == list(room.box_min)
        assert manifest["box"]["max"] == list(room.box_max)
        assert manifest["num_images"] > 0
        assert manifest["num_points"] > 0

    index = json.loads((chunks_dir / "chunks.json").read_text())
    assert {c["name"] for c in index["chunks"]} == {r.name for r in scene.rooms}


def test_chunk_manifest_records_through_doorway_observers(tmp_path):
    config = _config(tmp_path)
    scene = _scene(tmp_path)
    boxes_path = _write_boxes(tmp_path / "boxes.yaml", scene.rooms)

    chunk.run(_args(boxes=boxes_path), config)

    manifest = json.loads(
        (config.paths.output_dir / "chunks" / "room_b" / "manifest.json").read_text()
    )
    assert manifest["through_doorway_observers"] >= 1
    assert "room_a/doorway_000000.jpg" in manifest["images"]


def test_chunked_models_overlap_by_design(tmp_path):
    config = _config(tmp_path)
    scene = _scene(tmp_path)
    boxes_path = _write_boxes(tmp_path / "boxes.yaml", scene.rooms)

    chunk.run(_args(boxes=boxes_path), config)

    names = []
    for room in scene.rooms:
        model = read_model(config.paths.output_dir / "chunks" / room.name / "sparse" / "0")
        names.append({img.name for img in model.images.values()})

    # The doorway image belongs to both rooms; that overlap is what merge crops.
    assert names[0] & names[1]


def test_chunk_without_boxes_file_reports_how_to_get_one(tmp_path, caplog):
    config = _config(tmp_path)
    _scene(tmp_path)

    with caplog.at_level("ERROR"):
        exit_code = chunk.run(_args(boxes=tmp_path / "missing.yaml"), config)

    assert exit_code == 3
    assert any("--suggest" in r.message for r in caplog.records)


def test_chunk_with_no_model_is_nothing_to_do(tmp_path):
    config = _config(tmp_path)

    assert chunk.run(_args(), config) == 3


def test_box_matching_nothing_is_reported_as_failure(tmp_path, caplog):
    config = _config(tmp_path)
    _scene(tmp_path)
    save_boxes(
        tmp_path / "boxes.yaml",
        [BoundingBox("nowhere", (1000.0, 1000.0, 1000.0), (1001.0, 1001.0, 1001.0))],
    )

    with caplog.at_level("ERROR"):
        exit_code = chunk.run(_args(boxes=tmp_path / "boxes.yaml"), config)

    assert exit_code == 1
    assert any("no images select" in r.message for r in caplog.records)


def test_points_outside_every_box_are_reported(tmp_path, caplog):
    config = _config(tmp_path)
    scene = _scene(tmp_path)
    # Only chunk room A, leaving all of room B's points uncovered.
    save_boxes(
        tmp_path / "boxes.yaml",
        [BoundingBox(scene.rooms[0].name, scene.rooms[0].box_min, scene.rooms[0].box_max)],
    )

    with caplog.at_level("WARNING"):
        chunk.run(_args(boxes=tmp_path / "boxes.yaml"), config)

    assert any("fall outside every box" in r.message for r in caplog.records)


def test_min_points_in_box_flag_overrides_config(tmp_path):
    config = _config(tmp_path, min_points_in_box=1)
    scene = _scene(tmp_path)
    boxes_path = _write_boxes(tmp_path / "boxes.yaml", scene.rooms)

    chunk.run(_args(boxes=boxes_path, min_points_in_box=999), config)

    manifest = json.loads(
        (config.paths.output_dir / "chunks" / "room_b" / "manifest.json").read_text()
    )
    assert manifest["through_doorway_observers"] == 0
    assert manifest["min_points_in_box"] == 999


# ---------------------------------------------------------------------------
# --suggest
# ---------------------------------------------------------------------------


def test_suggest_proposes_one_box_per_room(tmp_path):
    config = _config(tmp_path, suggest_eps=5.0, suggest_min_samples=2)
    _scene(tmp_path)

    exit_code = chunk.run(_args(suggest=True, boxes=tmp_path / "suggested.yaml"), config)

    assert exit_code == 0
    boxes = load_boxes(tmp_path / "suggested.yaml")
    # The two room clusters are 20 units apart, well beyond eps.
    assert len(boxes) == 2


def test_suggested_boxes_cover_the_geometry_not_just_the_cameras(tmp_path):
    config = _config(tmp_path, suggest_eps=5.0, suggest_min_samples=2, suggest_percentile=0.0)
    scene = _scene(tmp_path)
    model = read_model(tmp_path / "output" / "sparse" / "0")

    chunk.run(_args(suggest=True, boxes=tmp_path / "suggested.yaml"), config)
    boxes = load_boxes(tmp_path / "suggested.yaml")

    # Every room's own points should land inside some suggested box; a box
    # drawn around camera positions alone would miss the walls.
    covered = np.zeros(len(model.points3d), dtype=bool)
    for box in boxes:
        covered |= box.contains(model.points3d.xyz)
    assert covered.sum() >= 0.9 * len(model.points3d)
    assert len(scene.rooms) == 2


def test_suggested_boxes_are_not_swallowed_by_a_doorway_observer(tmp_path):
    # The doorway camera clusters with room A but sees into room B. Sizing A's
    # box from everything its cameras observe would stretch it across both
    # rooms, making the two suggestions nearly identical and useless as chunks.
    config = _config(tmp_path, suggest_eps=5.0, suggest_min_samples=2)
    scene = _scene(tmp_path)

    chunk.run(_args(suggest=True, boxes=tmp_path / "suggested.yaml"), config)
    boxes = load_boxes(tmp_path / "suggested.yaml")

    assert len(boxes) == 2
    room_separation = abs(scene.rooms[1].box_min[0] - scene.rooms[0].box_min[0])
    for box in boxes:
        assert box.extent[0] < room_separation, (
            f"box {box.name!r} spans {box.extent[0]:.1f} in x, wider than the "
            f"{room_separation:.1f} gap between rooms -- it swallowed its neighbour"
        )

    # And the two boxes must not substantially overlap.
    lo = np.maximum(boxes[0].min, boxes[1].min)
    hi = np.minimum(boxes[0].max, boxes[1].max)
    assert np.any(hi < lo), "suggested boxes overlap; they should be one per room"


def test_suggest_with_kmeans(tmp_path):
    config = _config(tmp_path)
    _scene(tmp_path)

    exit_code = chunk.run(
        _args(suggest=True, suggest_method="kmeans", clusters=2,
              boxes=tmp_path / "suggested.yaml"),
        config,
    )

    assert exit_code == 0
    assert len(load_boxes(tmp_path / "suggested.yaml")) == 2


def test_suggested_yaml_is_directly_usable_by_chunk(tmp_path):
    config = _config(tmp_path, suggest_eps=5.0, suggest_min_samples=2)
    _scene(tmp_path)
    boxes_path = tmp_path / "suggested.yaml"

    assert chunk.run(_args(suggest=True, boxes=boxes_path), config) == 0
    assert chunk.run(_args(boxes=boxes_path), config) == 0

    index = json.loads((config.paths.output_dir / "chunks" / "chunks.json").read_text())
    assert len(index["chunks"]) == 2


def test_suggest_warns_when_a_box_exceeds_the_gaussian_budget(tmp_path, caplog):
    config = _config(
        tmp_path, suggest_eps=5.0, suggest_min_samples=2, target_gaussians_max=10
    )
    _scene(tmp_path)

    with caplog.at_level("WARNING"):
        chunk.run(_args(suggest=True, boxes=tmp_path / "suggested.yaml"), config)

    assert any("above the 10 target" in r.message for r in caplog.records)


def test_suggest_needs_enough_cameras(tmp_path, caplog):
    config = _config(tmp_path)
    tiny = build_healthy_model(num_images=1, num_points=4)
    write_model_bin(tmp_path / "output" / "sparse" / "0", tiny)

    with caplog.at_level("ERROR"):
        exit_code = chunk.run(_args(suggest=True, boxes=tmp_path / "s.yaml"), config)

    assert exit_code == 1
    assert any("at least 2 registered cameras" in r.message for r in caplog.records)
