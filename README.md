# splat-pipeline

A CLI pipeline for processing video and photos into a COLMAP reconstruction for 3D Gaussian Splatting.

Designed for large multi-room captures (e.g. an iPhone walkthrough video plus
supplemental stills): extract frames, select a well-distributed sharp subset,
organize by camera model, run COLMAP structure-from-motion, verify the
reconstruction, and optionally chunk/merge for very large scenes. Runs
entirely on CPU — no GPU dependencies.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Usage

```bash
splat <command> [--config config.yaml] [--log-level LEVEL]
```

Commands (currently stubs — logic not yet implemented):

```bash
splat extract   # sample frames from source video
splat select     # choose a sharp, well-distributed subset of images
splat organize   # lay out selected images by camera model for COLMAP
splat sfm        # run COLMAP structure-from-motion
splat verify      # sanity-check the reconstruction
splat chunk      # split a large reconstruction into overlapping chunks
splat merge      # recombine chunked results
```

Each run logs to the console and to a timestamped file under `logs/`
(configurable via `logging.log_dir`).

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
  fps: 2.0
  max_dimension: 3840

select:
  target_min: 4000
  target_max: 5500
  blur_threshold: 100.0

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
```

```bash
splat extract --config config.yaml
```

## Development

```bash
pytest -q
ruff check .
```
