import argparse
import csv
from pathlib import Path

from pipeline.colmap_utils import MissingBinaryError
from pipeline.commands import sfm
from pipeline.config import Paths, PipelineConfig, SfmConfig


class FakeRunner:
    """Records every command passed to run_streamed and returns scripted exit codes."""

    def __init__(self, returncodes):
        self.calls: list[list[str]] = []
        self._returncodes = list(returncodes)

    def __call__(self, command, logger):
        self.calls.append(command)
        if self._returncodes:
            return self._returncodes.pop(0)
        return 0


def _config(tmp_path, vocab_tree_exists=True, **sfm_kwargs):
    vocab_tree_path = tmp_path / "vocab_tree.bin"
    if vocab_tree_exists:
        vocab_tree_path.write_text("fake vocab tree data")

    return PipelineConfig(
        paths=Paths(
            input_video_dir=tmp_path / "input" / "video",
            input_photo_dir=tmp_path / "input" / "photos",
            workdir=tmp_path / "work",
            output_dir=tmp_path / "output",
        ),
        sfm=SfmConfig(**{"vocab_tree_path": vocab_tree_path, **sfm_kwargs}),
    )


def _make_images(config, video_count=2, photo_count=2):
    images_root = config.paths.output_dir / "images"
    for i in range(video_count):
        path = images_root / "video_4k" / f"clip_{i:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake jpg")
    for i in range(photo_count):
        path = images_root / "photos_48mp" / f"IMG_{i:04d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake jpg")
    return images_root


def _bypass_binary_check(monkeypatch, colmap="colmap", glomap="glomap"):
    binaries = {"colmap": colmap, "glomap": glomap}
    monkeypatch.setattr(sfm, "require_binary", lambda name: binaries[name])


def test_run_builds_expected_commands_for_all_stages(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _make_images(config)
    _bypass_binary_check(monkeypatch)
    fake_runner = FakeRunner([0, 0, 0, 0])
    monkeypatch.setattr(sfm, "run_streamed", fake_runner)

    exit_code = sfm.run(argparse.Namespace(), config)

    assert exit_code == 0
    assert len(fake_runner.calls) == 4

    feature_cmd, sequential_cmd, vocab_cmd, glomap_cmd = fake_runner.calls

    assert feature_cmd[:2] == ["colmap", "feature_extractor"]
    assert "--database_path" in feature_cmd
    assert "--image_path" in feature_cmd
    db_path = feature_cmd[feature_cmd.index("--database_path") + 1]
    image_path = feature_cmd[feature_cmd.index("--image_path") + 1]
    assert db_path == str(config.paths.workdir / "sfm" / "database.db")
    assert image_path == str(config.paths.output_dir / "images")
    assert "--ImageReader.camera_model" in feature_cmd
    assert feature_cmd[feature_cmd.index("--ImageReader.camera_model") + 1] == "OPENCV"
    assert feature_cmd[feature_cmd.index("--ImageReader.single_camera_per_folder") + 1] == "1"
    assert feature_cmd[feature_cmd.index("--SiftExtraction.use_gpu") + 1] == "0"

    assert sequential_cmd[:2] == ["colmap", "sequential_matcher"]
    assert sequential_cmd[sequential_cmd.index("--SequentialMatching.loop_detection") + 1] == "1"
    assert sequential_cmd[sequential_cmd.index("--SequentialMatching.vocab_tree_path") + 1] == (
        str(config.sfm.vocab_tree_path)
    )

    assert vocab_cmd[:2] == ["colmap", "vocab_tree_matcher"]
    assert vocab_cmd[vocab_cmd.index("--VocabTreeMatching.vocab_tree_path") + 1] == (
        str(config.sfm.vocab_tree_path)
    )
    match_list_path = vocab_cmd[vocab_cmd.index("--VocabTreeMatching.match_list_path") + 1]
    match_list_contents = Path(match_list_path).read_text()
    assert "photos_48mp/IMG_0000.jpg" in match_list_contents
    assert "photos_48mp/IMG_0001.jpg" in match_list_contents
    assert "video_4k" not in match_list_contents

    assert glomap_cmd[:2] == ["glomap", "mapper"]
    glomap_image_path = glomap_cmd[glomap_cmd.index("--image_path") + 1]
    assert glomap_image_path == str(config.paths.output_dir / "images")

    for stage in ["feature_extractor", "sequential_matcher", "vocab_tree_matcher", "glomap_mapper"]:
        assert (config.paths.workdir / "sfm" / f".{stage}.done").exists()


def test_resume_skips_completed_stages(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _make_images(config)
    _bypass_binary_check(monkeypatch)
    sfm_dir = config.paths.workdir / "sfm"
    sfm_dir.mkdir(parents=True)
    (sfm_dir / ".feature_extractor.done").write_text("1.0\n")
    (sfm_dir / ".sequential_matcher.done").write_text("1.0\n")

    fake_runner = FakeRunner([0, 0])
    monkeypatch.setattr(sfm, "run_streamed", fake_runner)

    exit_code = sfm.run(argparse.Namespace(), config)

    assert exit_code == 0
    assert len(fake_runner.calls) == 2
    assert fake_runner.calls[0][1] == "vocab_tree_matcher"
    assert fake_runner.calls[1][1] == "mapper"


def test_gpu_failure_falls_back_to_cpu(tmp_path, monkeypatch):
    config = _config(tmp_path, use_gpu=True)
    _make_images(config)
    _bypass_binary_check(monkeypatch)
    fake_runner = FakeRunner([1, 0, 0, 0, 0])
    monkeypatch.setattr(sfm, "run_streamed", fake_runner)

    exit_code = sfm.run(argparse.Namespace(), config)

    assert exit_code == 0
    assert len(fake_runner.calls) == 5

    first_attempt, retry, *_rest = fake_runner.calls
    assert first_attempt[:2] == ["colmap", "feature_extractor"]
    assert first_attempt[first_attempt.index("--SiftExtraction.use_gpu") + 1] == "1"
    assert retry[:2] == ["colmap", "feature_extractor"]
    assert retry[retry.index("--SiftExtraction.use_gpu") + 1] == "0"

    marker = config.paths.workdir / "sfm" / ".feature_extractor.done"
    assert marker.exists()


def test_gpu_failure_with_no_cpu_recovery_fails_stage(tmp_path, monkeypatch):
    config = _config(tmp_path, use_gpu=True)
    _make_images(config)
    _bypass_binary_check(monkeypatch)
    fake_runner = FakeRunner([1, 1])
    monkeypatch.setattr(sfm, "run_streamed", fake_runner)

    exit_code = sfm.run(argparse.Namespace(), config)

    assert exit_code == 1
    assert len(fake_runner.calls) == 2
    marker = config.paths.workdir / "sfm" / ".feature_extractor.done"
    assert not marker.exists()

    timing_path = config.paths.workdir / "sfm" / "timing_summary.csv"
    with timing_path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["stage"] == "feature_extractor"
    assert rows[0]["status"] == "failed"


def test_missing_binary_fails_loudly_and_stops_pipeline(tmp_path, monkeypatch, caplog):
    config = _config(tmp_path)
    _make_images(config)

    def _raise_missing(name):
        raise MissingBinaryError(f"Required binary '{name}' was not found on PATH.")

    monkeypatch.setattr(sfm, "require_binary", _raise_missing)
    fake_runner = FakeRunner([])
    monkeypatch.setattr(sfm, "run_streamed", fake_runner)

    with caplog.at_level("ERROR"):
        exit_code = sfm.run(argparse.Namespace(), config)

    assert exit_code == 1
    assert len(fake_runner.calls) == 0
    assert any("was not found on PATH" in r.message for r in caplog.records)


def test_missing_vocab_tree_fails_loudly(tmp_path, monkeypatch, caplog):
    config = _config(tmp_path, vocab_tree_exists=False)
    _make_images(config)
    _bypass_binary_check(monkeypatch)
    fake_runner = FakeRunner([0])
    monkeypatch.setattr(sfm, "run_streamed", fake_runner)

    with caplog.at_level("ERROR"):
        exit_code = sfm.run(argparse.Namespace(), config)

    assert exit_code == 1
    # feature_extractor (no vocab tree needed) succeeds; sequential_matcher fails before
    # ever invoking run_streamed for that stage.
    assert len(fake_runner.calls) == 1
    assert any("Vocab tree file not found" in r.message for r in caplog.records)


def test_vocab_tree_matcher_skips_when_no_photos(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _make_images(config, video_count=3, photo_count=0)
    _bypass_binary_check(monkeypatch)
    fake_runner = FakeRunner([0, 0, 0])
    monkeypatch.setattr(sfm, "run_streamed", fake_runner)

    exit_code = sfm.run(argparse.Namespace(), config)

    assert exit_code == 0
    # feature_extractor, sequential_matcher, glomap_mapper call run_streamed;
    # vocab_tree_matcher has no photos so it never does.
    assert len(fake_runner.calls) == 3
    called_binaries = [(c[0], c[1]) for c in fake_runner.calls]
    assert ("colmap", "vocab_tree_matcher") not in called_binaries
    marker = config.paths.workdir / "sfm" / ".vocab_tree_matcher.done"
    assert marker.exists()
