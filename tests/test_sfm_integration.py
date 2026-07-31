"""Real (non-mocked) integration test that drives actual colmap/glomap binaries.

Only runs if colmap is present on PATH; glomap is checked separately since it's
far less commonly installed, and the final mapper stage is optional coverage.
"""

import argparse
import logging
import shutil

import cv2
import numpy as np
import pytest

from pipeline.colmap_utils import run_streamed
from pipeline.commands import sfm
from pipeline.config import Paths, PipelineConfig, SfmConfig

COLMAP_PATH = shutil.which("colmap")
GLOMAP_PATH = shutil.which("glomap")


def _make_textured_canvas():
    rng = np.random.default_rng(0)
    canvas_size = (720, 960)  # height, width
    canvas = rng.integers(0, 255, (*canvas_size, 3), dtype=np.uint8)
    for _ in range(60):
        center = (int(rng.integers(0, canvas_size[1])), int(rng.integers(0, canvas_size[0])))
        radius = int(rng.integers(5, 40))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(canvas, center, radius, color, -1)
    return canvas


def _make_overlapping_crops(canvas, images_root, group_name, count=6):
    """Generate `count` overlapping crops of `canvas`.

    This gives COLMAP genuine (if tiny) inter-image correspondences to find,
    simulating a slowly panning camera over a textured static scene.
    """
    canvas_size = canvas.shape[:2]
    crop_size = (480, 360)  # width, height
    max_x = canvas_size[1] - crop_size[0]
    max_y = canvas_size[0] - crop_size[1]
    step_x = max_x // max(count - 1, 1)

    group_dir = images_root / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        x0 = min(i * step_x, max_x)
        y0 = max_y // 2
        crop = canvas[y0 : y0 + crop_size[1], x0 : x0 + crop_size[0]]
        cv2.imwrite(str(group_dir / f"clip_{i:06d}.jpg"), crop)


def _build_vocab_tree(database_path, vocab_tree_path):
    command = [
        COLMAP_PATH, "vocab_tree_builder",
        "--database_path", str(database_path),
        "--vocab_tree_path", str(vocab_tree_path),
        "--num_visual_words", "16",
        "--branching", "2",
        "--num_iterations", "1",
    ]
    exit_code = run_streamed(command, logging.getLogger("test.vocab_tree_builder"))
    assert exit_code == 0, "colmap vocab_tree_builder failed"


@pytest.mark.colmap_integration
@pytest.mark.skipif(COLMAP_PATH is None, reason="colmap binary not found on PATH")
def test_sfm_runs_real_colmap_on_tiny_synthetic_image_set(tmp_path):
    config = PipelineConfig(
        paths=Paths(
            input_video_dir=tmp_path / "input" / "video",
            input_photo_dir=tmp_path / "input" / "photos",
            workdir=tmp_path / "work",
            output_dir=tmp_path / "output",
        ),
        sfm=SfmConfig(vocab_tree_path=tmp_path / "vocab_tree.bin"),
    )

    paths = sfm._sfm_paths(config)
    canvas = _make_textured_canvas()
    _make_overlapping_crops(canvas, paths.images_root, "video_test", count=6)
    # A "photos_" group so vocab_tree_matcher's real subprocess call is
    # actually exercised, not just its no-photos skip path.
    _make_overlapping_crops(canvas, paths.images_root, "photos_test", count=2)

    # Pre-run feature_extractor for real so we have a populated database to
    # build a (tiny, self-contained) vocab tree from, then mark that stage as
    # already complete so `sfm.run()` resumes from sequential_matcher using
    # the same database -- this exercises real resume behavior too.
    assert sfm._run_feature_extractor(config, paths)
    _build_vocab_tree(paths.database_path, config.sfm.vocab_tree_path)
    sfm._marker_path(paths, "feature_extractor").parent.mkdir(parents=True, exist_ok=True)
    sfm._marker_path(paths, "feature_extractor").write_text("0.0\n")

    exit_code = sfm.run(argparse.Namespace(), config)

    assert sfm._marker_path(paths, "sequential_matcher").exists()
    assert sfm._marker_path(paths, "vocab_tree_matcher").exists()
    # Confirms the real vocab_tree_matcher subprocess ran against actual photo
    # images, rather than hitting its no-photos skip path.
    assert "photos_test" in paths.match_list_path.read_text()

    if GLOMAP_PATH is not None:
        assert exit_code == 0
        assert sfm._marker_path(paths, "glomap_mapper").exists()
        assert any(paths.sparse_output_dir.rglob("*"))
    else:
        assert exit_code == 1
        assert not sfm._marker_path(paths, "glomap_mapper").exists()
