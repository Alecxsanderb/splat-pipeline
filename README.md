# splat-pipeline

A CLI pipeline for processing video and photos into a COLMAP reconstruction for 3D Gaussian Splatting.

Designed for large multi-room captures (e.g. an iPhone walkthrough video plus
supplemental stills): extract frames, select a well-distributed sharp subset,
organize by camera model, run COLMAP structure-from-motion, verify the
reconstruction, and optionally chunk/merge for very large scenes. Runs
entirely on CPU — no GPU dependencies.

## Prerequisites

- Python 3.11+
- `ffmpeg` / `ffprobe` on `PATH` (used by `extract`; install via your system
  package manager, e.g. `apt install ffmpeg`)

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
splat <command> [--config config.yaml] [--log-level LEVEL]
```

Commands:

```bash
splat extract    # sample frames from source video (implemented)
splat select     # choose a sharp, well-distributed subset of images (implemented)
splat organize   # lay out selected images by camera model for COLMAP (stub)
splat sfm        # run COLMAP structure-from-motion (stub)
splat verify     # sanity-check the reconstruction (stub)
splat chunk      # split a large reconstruction into overlapping chunks (stub)
splat merge      # recombine chunked results (stub)
```

Each run logs to the console and to a timestamped file under `logs/`
(configurable via `logging.log_dir`).

### extract

Recursively finds video files under `paths.input_video_dir` and extracts
JPEG frames via `ffmpeg` at `extract.fps` (default 4 fps), writing them to
`<workdir>/frames/<relative-dir>/<clip-stem>_%06d.jpg` — the source clip name
is preserved in the output filenames. Clips are processed in parallel across
a worker pool (`extract.workers`). Clips that already have a completed
extraction (marked by a `.<clip-stem>.done` file) are skipped, so a
re-run after a partial failure only does the remaining work.

```bash
splat extract --config config.yaml            # extract all clips
splat extract --config config.yaml --dry-run  # estimate frame counts (via ffprobe) without extracting
```

### select

Scores every extracted frame's sharpness using the variance of the Laplacian
(OpenCV), then applies windowed selection: within each non-overlapping window
of `select.window_size` consecutive frames per clip (default 4, i.e. ~1s at
4fps), only the sharpest frame is kept. A frame is only kept if it is also
above `select.blur_threshold` in absolute terms, so a window that's uniformly
blurry (e.g. a fast pan) still gets dropped even though it "won" its window.

Every scored frame — kept or dropped — is written to
`<workdir>/select/frame_scores.csv` with its sharpness score and decision, so
thresholds can be tuned by inspecting the CSV without re-running `extract`.
Frames that are kept are copied to `<workdir>/selected/<relative-dir>/`.

```bash
splat select --config config.yaml            # score, report, and copy selected frames
splat select --config config.yaml --dry-run  # score and write the report only, skip copying
```

### Per-source overrides

Global sweep footage typically needs a lower fps / larger window than
detail passes of a single room. `overrides` lets you set `fps` (used by
`extract`) and/or `window_size` (used by `select`) per source subdirectory,
matched against the longest matching path prefix relative to
`paths.input_video_dir`:

```yaml
overrides:
  - path: global_sweeps
    fps: 2.0
  - path: room_kitchen/detail
    fps: 6.0
    window_size: 6
```

## Configuration

All stages read from a single `PipelineConfig`, loaded from defaults or from
a YAML file passed via `--config`. Any subset of fields can be overridden;
everything else falls back to its default.

```yaml
paths:
  input_video_dir: input/video
  input_photo_dir: input/photos
  workdir: work
  output_dir: output

extract:
  fps: 4.0
  max_dimension: 3840
  workers: 4

select:
  target_min: 4000
  target_max: 5500
  blur_threshold: 100.0
  window_size: 4
  workers: 4

organize:
  group_by_camera_model: true

sfm:
  matcher: sequential
  camera_model: OPENCV

chunk:
  max_images_per_chunk: 1500
  overlap: 100

logging:
  level: INFO
  log_dir: logs

overrides:
  - path: global_sweeps
    fps: 2.0
```

## Development

```bash
pytest -q
ruff check .
```

Tests for `extract` and `select` generate real synthetic videos with
`ffmpeg` (some frames deliberately blurred) and run the actual `ffmpeg`/
OpenCV pipeline end-to-end — no mocking — so `ffmpeg` must be installed to
run the full test suite.
