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
- `colmap` on `PATH` (used by `sfm`; e.g. `apt install colmap` — a CPU-only
  build works fine)
- `glomap` on `PATH` (used by `sfm`'s final mapping stage; see
  https://github.com/colmap/glomap — not packaged by most distros, build
  from source)
- A COLMAP vocabulary tree file (used by `sfm`'s matching stages; download a
  prebuilt one from https://demuc.de/colmap/vocab_tree_data/, or build a
  custom one with `colmap vocab_tree_builder` from your own extracted
  features), pointed to by `sfm.vocab_tree_path`

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
splat organize   # lay out selected images by camera model for COLMAP (implemented)
splat sfm        # run COLMAP + GLOMAP structure-from-motion (implemented)
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

### organize

Copies selected video frames (from `<workdir>/selected/`) and photos (from
`paths.input_photo_dir`) into `<output_dir>/images/<group>/`, one subfolder
per "camera model" — in practice, a group per (video or photo) source type
and resolution bucket, e.g. `video_4k/`, `video_1080p/`, `photos_48mp/`.
Grouping by resolution matters because COLMAP's `feature_extractor` is run
with `--ImageReader.single_camera_per_folder 1`, i.e. it assumes one set of
camera intrinsics per folder — mixing resolutions in one folder would give
COLMAP the wrong intrinsics for some images. Set `organize.group_by_camera_model:
false` to instead copy everything into a flat `video/` and `photos/` split.

Copies are skipped if the destination file already exists, so `organize` is
resumable. After copying, it logs a per-group image count and flags any
group with fewer than `organize.suspicious_min_images` (default 20) images
as suspicious — usually a sign that a room or clip didn't make it through
`select`.

```bash
splat organize --config config.yaml
```

### sfm

Drives COLMAP and GLOMAP as subprocesses against the `<output_dir>/images/`
layout produced by `organize`, in four stages:

1. `colmap feature_extractor` (camera model `sfm.camera_model`, one camera
   per folder). Uses `sfm.use_gpu` if set, and automatically retries on CPU
   if the GPU attempt fails.
2. `colmap sequential_matcher` with loop detection, using `sfm.vocab_tree_path`.
3. `colmap vocab_tree_matcher`, restricted (via a generated match list) to
   the non-sequential `photos_*` images — video frames are already covered
   by sequential matching.
4. `glomap mapper`, writing the sparse reconstruction to `<output_dir>/sparse/`.

Each stage is resumable: a `.<stage>.done` marker file under
`<workdir>/sfm/` causes a re-run to skip that stage entirely. Subprocess
output is streamed line-by-line to both the console and the run log as it
happens. If `colmap` or `glomap` isn't on `PATH`, or the vocab tree file is
missing, the stage fails immediately with a clear error and the pipeline
stops (no later stage runs against an incomplete earlier one). A timing
summary for every stage (including skipped ones) is written to
`<workdir>/sfm/timing_summary.csv` and logged at the end of the run.

```bash
splat sfm --config config.yaml
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
  photo_extensions: [".jpg", ".jpeg", ".png"]
  suspicious_min_images: 20

sfm:
  matcher: sequential
  camera_model: OPENCV
  use_gpu: false
  vocab_tree_path: vocab_tree.bin

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

`sfm` is tested at two levels: command construction, resume logic, and the
GPU→CPU fallback are covered by mocking only the subprocess-execution layer
(so real argv lists are still verified). A separate test marked
`colmap_integration` drives real `colmap` (and `glomap`, if present) against
a tiny synthetic image set with genuine inter-image overlap; it's skipped
automatically if `colmap` isn't on `PATH`.

```bash
pytest -q -m colmap_integration   # run just the real-COLMAP integration test
```
