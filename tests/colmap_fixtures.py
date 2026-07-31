"""Writers and builders for synthetic COLMAP sparse models.

IMPORTANT: this module must not import anything from `pipeline.colmap_model`.
It deliberately re-declares its own struct formats and its own invalid-point
sentinel. If the writer borrowed the reader's constants, a typo in a format
string would be invisible to every round-trip test, because both sides would be
wrong in exactly the same way. The duplication is the point.

Both binary and text writers are provided. The text writers are load-bearing,
not a convenience: they are the input side of the real-COLMAP validation, which
feeds text through `colmap model_converter` and reads back the binary COLMAP
itself produced.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Re-declared on purpose -- see the module docstring.
FIXTURE_INVALID_POINT3D_ID = 2**64 - 1

_U64 = struct.Struct("<Q")
_CAMERA_HEADER = struct.Struct("<IiQQ")
_IMAGE_HEADER = struct.Struct("<I7dI")
_POINT2D = struct.Struct("<ddQ")
_POINT3D_HEADER = struct.Struct("<Q3d3BdQ")
_TRACK_ELEM = struct.Struct("<II")

# (name, num_params) for COLMAP 3.9.x, ids 0..10.
CAMERA_MODEL_IDS: dict[str, int] = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}

CAMERA_MODEL_NUM_PARAMS: dict[str, int] = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}

# Exactly-representable binary fractions, so a text round-trip through COLMAP
# is lossless and tests can assert exact equality instead of tolerances.
_DEFAULT_PARAMS: dict[str, tuple[float, ...]] = {
    "SIMPLE_PINHOLE": (1200.0, 960.0, 540.0),
    "PINHOLE": (1200.0, 1200.0, 960.0, 540.0),
    "SIMPLE_RADIAL": (1200.0, 960.0, 540.0, 0.0),
    "RADIAL": (1200.0, 960.0, 540.0, 0.0, 0.0),
    "OPENCV": (1200.0, 1200.0, 960.0, 540.0, 0.0, 0.0, 0.0, 0.0),
    "OPENCV_FISHEYE": (1200.0, 1200.0, 960.0, 540.0, 0.0, 0.0, 0.0, 0.0),
    "FULL_OPENCV": (
        1200.0, 1200.0, 960.0, 540.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ),
    "FOV": (1200.0, 1200.0, 960.0, 540.0, 0.5),
    "SIMPLE_RADIAL_FISHEYE": (1200.0, 960.0, 540.0, 0.0),
    "RADIAL_FISHEYE": (1200.0, 960.0, 540.0, 0.0, 0.0),
    "THIN_PRISM_FISHEYE": (
        1200.0, 1200.0, 960.0, 540.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ),
}


@dataclass
class FixtureCamera:
    camera_id: int = 1
    model: str | int = "OPENCV"
    width: int = 1920
    height: int = 1080
    params: tuple[float, ...] | None = None

    @property
    def model_id(self) -> int:
        if isinstance(self.model, int):
            return self.model
        return CAMERA_MODEL_IDS[self.model]

    @property
    def model_name(self) -> str:
        if isinstance(self.model, int):
            for name, mid in CAMERA_MODEL_IDS.items():
                if mid == self.model:
                    return name
            raise KeyError(f"no camera model name for id {self.model}")
        return self.model

    def resolved_params(self) -> tuple[float, ...]:
        if self.params is not None:
            return self.params
        return _DEFAULT_PARAMS[self.model_name]


@dataclass
class FixtureImage:
    image_id: int
    name: str
    camera_id: int = 1
    qvec: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    tvec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # (x, y, point3D id or None for "no 3D point")
    points2d: list[tuple[float, float, int | None]] = field(default_factory=list)


@dataclass
class FixturePoint3D:
    point3d_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int] = (128, 128, 128)
    error: float = 0.5
    track: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class FixtureModel:
    cameras: list[FixtureCamera] = field(default_factory=list)
    images: list[FixtureImage] = field(default_factory=list)
    points3d: list[FixturePoint3D] = field(default_factory=list)


# --------------------------------------------------------------------------
# Binary writers
# --------------------------------------------------------------------------


def write_cameras_bin(path: Path, cameras: Sequence[FixtureCamera]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [_U64.pack(len(cameras))]
    for camera in cameras:
        chunks.append(
            _CAMERA_HEADER.pack(camera.camera_id, camera.model_id, camera.width, camera.height)
        )
        params = camera.resolved_params()
        chunks.append(struct.pack(f"<{len(params)}d", *params))
    path.write_bytes(b"".join(chunks))


def write_images_bin(path: Path, images: Sequence[FixtureImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [_U64.pack(len(images))]
    for image in images:
        chunks.append(
            _IMAGE_HEADER.pack(image.image_id, *image.qvec, *image.tvec, image.camera_id)
        )
        chunks.append(image.name.encode("utf-8", errors="surrogateescape") + b"\x00")
        chunks.append(_U64.pack(len(image.points2d)))
        for x, y, point3d_id in image.points2d:
            resolved = FIXTURE_INVALID_POINT3D_ID if point3d_id is None else point3d_id
            chunks.append(_POINT2D.pack(x, y, resolved))
    path.write_bytes(b"".join(chunks))


def write_points3d_bin(path: Path, points: Sequence[FixturePoint3D]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [_U64.pack(len(points))]
    for point in points:
        chunks.append(
            _POINT3D_HEADER.pack(
                point.point3d_id, *point.xyz, *point.rgb, point.error, len(point.track)
            )
        )
        for image_id, point2d_idx in point.track:
            chunks.append(_TRACK_ELEM.pack(image_id, point2d_idx))
    path.write_bytes(b"".join(chunks))


def write_model_bin(model_dir: Path, model: FixtureModel) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    write_cameras_bin(model_dir / "cameras.bin", model.cameras)
    write_images_bin(model_dir / "images.bin", model.images)
    write_points3d_bin(model_dir / "points3D.bin", model.points3d)
    return model_dir


# --------------------------------------------------------------------------
# Text writers (input side of the real-COLMAP validation)
# --------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return repr(float(value))


def write_cameras_txt(path: Path, cameras: Sequence[FixtureCamera]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Camera list with one line of data per camera:",
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
             f"# Number of cameras: {len(cameras)}"]
    for camera in cameras:
        params = " ".join(_fmt(p) for p in camera.resolved_params())
        lines.append(
            f"{camera.camera_id} {camera.model_name} {camera.width} {camera.height} {params}"
        )
    path.write_text("\n".join(lines) + "\n")


def write_images_txt(path: Path, images: Sequence[FixtureImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Image list with two lines of data per image:",
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
             "#   POINTS2D[] as (X, Y, POINT3D_ID)",
             f"# Number of images: {len(images)}"]
    for image in images:
        qvec = " ".join(_fmt(v) for v in image.qvec)
        tvec = " ".join(_fmt(v) for v in image.tvec)
        lines.append(f"{image.image_id} {qvec} {tvec} {image.camera_id} {image.name}")
        # COLMAP reads exactly two lines per image, so the second line must be
        # emitted even when empty or every following image is misparsed.
        parts = []
        for x, y, point3d_id in image.points2d:
            parts.append(f"{_fmt(x)} {_fmt(y)} {-1 if point3d_id is None else point3d_id}")
        lines.append(" ".join(parts))
    path.write_text("\n".join(lines) + "\n")


def write_points3d_txt(path: Path, points: Sequence[FixturePoint3D]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 3D point list with one line of data per point:",
             "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
             f"# Number of points: {len(points)}"]
    for point in points:
        xyz = " ".join(_fmt(v) for v in point.xyz)
        rgb = " ".join(str(int(v)) for v in point.rgb)
        track = " ".join(f"{img} {idx}" for img, idx in point.track)
        lines.append(f"{point.point3d_id} {xyz} {rgb} {_fmt(point.error)} {track}")
    path.write_text("\n".join(lines) + "\n")


def write_model_txt(model_dir: Path, model: FixtureModel) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    write_cameras_txt(model_dir / "cameras.txt", model.cameras)
    write_images_txt(model_dir / "images.txt", model.images)
    write_points3d_txt(model_dir / "points3D.txt", model.points3d)
    return model_dir


# --------------------------------------------------------------------------
# Corruption helpers, for the error paths real COLMAP cannot produce for us
# --------------------------------------------------------------------------


def truncate_file(path: Path, keep_bytes: int) -> None:
    data = path.read_bytes()
    path.write_bytes(data[:keep_bytes])


def append_garbage(path: Path, nbytes: int = 7) -> None:
    with path.open("ab") as handle:
        handle.write(b"\xde" * nbytes)


def set_leading_count(path: Path, count: int) -> None:
    data = bytearray(path.read_bytes())
    data[0:8] = _U64.pack(count)
    path.write_bytes(bytes(data))


def patch_bytes(path: Path, offset: int, payload: bytes) -> None:
    data = bytearray(path.read_bytes())
    data[offset : offset + len(payload)] = payload
    path.write_bytes(bytes(data))


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def make_consistent(model: FixtureModel) -> FixtureModel:
    """Rewrite each image's point3D id column to agree with the points3D tracks.

    Builders call this so they cannot accidentally emit a model that trips the
    reader's bidirectional cross-check; the tests that target that check
    deliberately skip it.
    """
    by_id = {image.image_id: image for image in model.images}
    for image in model.images:
        image.points2d = [(x, y, None) for x, y, _ in image.points2d]
    for point in model.points3d:
        for image_id, idx in point.track:
            image = by_id[image_id]
            x, y, _ = image.points2d[idx]
            image.points2d[idx] = (x, y, point.point3d_id)
    return model


def _arc_pose(index: int, count: int, radius: float = 4.0) -> tuple[
    tuple[float, float, float, float], tuple[float, float, float]
]:
    """Camera pose looking inward from a point on a circular arc."""
    angle = (index / max(count, 1)) * (np.pi / 2)
    center = np.array([radius * np.cos(angle), 0.25 * index, radius * np.sin(angle)])
    forward = -center / np.linalg.norm(center)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    rot = np.stack([right, true_up, forward])  # world -> camera
    qvec = _rotmat_to_qvec(rot)
    tvec = -rot @ center
    return qvec, (float(tvec[0]), float(tvec[1]), float(tvec[2]))


def _rotmat_to_qvec(rot: np.ndarray) -> tuple[float, float, float, float]:
    trace = np.trace(rot)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def build_healthy_model(
    *,
    num_images: int = 12,
    num_points: int = 120,
    points2d_per_image: int = 40,
    track_length: int = 5,
    camera_model: str = "OPENCV",
    width: int = 1920,
    height: int = 1080,
    name_template: str = "video_test/frame_{index:06d}.jpg",
    error: float = 0.5,
    first_image_id: int = 1,
    first_point_id: int = 1,
    seed: int = 0,
) -> FixtureModel:
    """A well-connected model: overlapping tracks across consecutive images."""
    rng = np.random.default_rng(seed)
    cameras = [FixtureCamera(camera_id=1, model=camera_model, width=width, height=height)]

    images = []
    for i in range(num_images):
        qvec, tvec = _arc_pose(i, num_images)
        images.append(
            FixtureImage(
                image_id=first_image_id + i,
                name=name_template.format(index=i),
                camera_id=1,
                qvec=qvec,
                tvec=tvec,
                points2d=[
                    (float(rng.integers(0, width)), float(rng.integers(0, height)), None)
                    for _ in range(points2d_per_image)
                ],
            )
        )

    # Each track covers `track_length` consecutive images, sliding along the
    # sequence, which is what makes the co-visibility graph connected.
    points = []
    next_slot = {img.image_id: 0 for img in images}
    for p in range(num_points):
        start = p % max(num_images - track_length + 1, 1)
        track = []
        for offset in range(min(track_length, num_images)):
            image_id = first_image_id + start + offset
            if image_id > first_image_id + num_images - 1:
                break
            slot = next_slot[image_id]
            if slot >= points2d_per_image:
                continue
            next_slot[image_id] = slot + 1
            track.append((image_id, slot))
        if len(track) < 2:
            continue
        points.append(
            FixturePoint3D(
                point3d_id=first_point_id + p,
                xyz=(float(p % 10), float((p // 10) % 10), float(p % 7)),
                rgb=(128, 128, 128),
                error=error,
                track=track,
            )
        )

    return make_consistent(FixtureModel(cameras=cameras, images=images, points3d=points))


def build_fragmented_model(
    *,
    component_sizes: Sequence[int] = (8, 6),
    component_name_templates: Sequence[str] | None = None,
    points_per_component: int = 60,
    bridge_tracks: int = 0,
    bridge_track_length: int = 4,
    component_separation: float = 100.0,
    **healthy_kwargs,
) -> FixtureModel:
    """A model whose co-visibility graph splits into several components.

    Every track stays inside one component, so the components share no 3D
    points at all. `bridge_tracks` adds tracks that span the first two
    components, which is how a *weakly* connected scene is simulated -- the
    difference between "disconnected" and "weakly connected" is exactly the
    threshold behaviour under test.
    """
    templates = component_name_templates or [
        f"video_part{i}/frame_{{index:06d}}.jpg" for i in range(len(component_sizes))
    ]

    cameras = [FixtureCamera(camera_id=1, model="OPENCV")]
    images: list[FixtureImage] = []
    points: list[FixturePoint3D] = []

    next_image_id = 1
    next_point_id = 1
    component_image_ids: list[list[int]] = []

    for comp_index, size in enumerate(component_sizes):
        part = build_healthy_model(
            num_images=size,
            num_points=points_per_component,
            name_template=templates[comp_index],
            first_image_id=next_image_id,
            first_point_id=next_point_id,
            seed=comp_index,
            **healthy_kwargs,
        )
        # Push each component far apart in world space so the spatial fallback
        # has something meaningful to measure.
        shift = comp_index * component_separation
        for image in part.images:
            image.tvec = (image.tvec[0] + shift, image.tvec[1], image.tvec[2])
        images.extend(part.images)
        points.extend(part.points3d)
        component_image_ids.append([img.image_id for img in part.images])
        next_image_id += size
        next_point_id += len(part.points3d) + points_per_component

    if bridge_tracks and len(component_image_ids) >= 2:
        by_id = {img.image_id: img for img in images}
        used = {img.image_id: sum(1 for p in img.points2d if p[2] is not None) for img in images}
        for b in range(bridge_tracks):
            track = []
            for comp in (0, 1):
                ids = component_image_ids[comp]
                for k in range(bridge_track_length // 2):
                    image_id = ids[(b + k) % len(ids)]
                    slot = used[image_id]
                    if slot >= len(by_id[image_id].points2d):
                        continue
                    used[image_id] = slot + 1
                    track.append((image_id, slot))
            if len(track) >= 2:
                points.append(
                    FixturePoint3D(
                        point3d_id=next_point_id + b,
                        xyz=(float(b), 0.0, 0.0),
                        error=0.5,
                        track=track,
                    )
                )

    return make_consistent(FixtureModel(cameras=cameras, images=images, points3d=points))


def build_poorly_registered_model(
    *,
    num_images: int = 4,
    num_points: int = 30,
    **kwargs,
) -> FixtureModel:
    """A small model, used to simulate most input images failing to register."""
    return build_healthy_model(num_images=num_images, num_points=num_points, **kwargs)
