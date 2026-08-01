"""Hand-rolled reader and writer for COLMAP sparse-model binaries.

Reads and writes `cameras.bin`, `images.bin` and `points3D.bin` without
depending on pycolmap, and builds self-consistent sub-models (used by `chunk`).
Correctness is the priority here: `verify` reports on a capture that cannot be
re-shot, so this module fails loudly and specifically rather than ever
returning partial or plausible-but-wrong data.

Binary layout notes worth knowing before editing:

* Everything is little-endian and packed -- there is no alignment padding.
  The points3D record header is genuinely 51 bytes (three RGB bytes wedged
  between 8-byte fields); a numpy dtype with ``align=True`` would silently
  round it to 56 and corrupt every record.
* ``model_id`` in cameras.bin is a *signed* int32 (COLMAP writes a C ``int``).
* A ``point3D_id`` of 2**64-1 means "this 2D feature has no 3D point". The
  text format spells the same thing ``-1``.
* The leading count in images.bin is the number of *registered* images --
  COLMAP omits unregistered ones entirely, so this file can never tell you how
  many images were originally offered to the reconstruction.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ColmapModelError(RuntimeError):
    """Base class for every failure reading a COLMAP sparse model."""


class MissingModelFileError(ColmapModelError):
    """A required model file or directory is absent."""


class TruncatedFileError(ColmapModelError):
    """A model file ended before the data it declared."""


class UnknownCameraModelError(ColmapModelError):
    """A camera record used a camera model id this reader does not know."""


class InconsistentModelError(ColmapModelError):
    """The model parsed, but its contents contradict each other."""


_U64 = struct.Struct("<Q")
_CAMERA_HEADER = struct.Struct("<IiQQ")  # camera_id, model_id (signed), width, height
_IMAGE_HEADER = struct.Struct("<I7dI")  # image_id, qw qx qy qz, tx ty tz, camera_id
_POINT3D_HEADER = struct.Struct("<Q3d3BdQ")  # id, xyz, rgb, error, track_length

_POINT2D_DTYPE = np.dtype([("xy", "<f8", (2,)), ("point3D_id", "<u8")])

# A mis-sized format string here would corrupt everything downstream, so pin
# the sizes at import rather than discovering it against a real capture.
assert _CAMERA_HEADER.size == 24, _CAMERA_HEADER.size
assert _IMAGE_HEADER.size == 64, _IMAGE_HEADER.size
assert _POINT3D_HEADER.size == 51, _POINT3D_HEADER.size
assert _POINT2D_DTYPE.itemsize == 24, _POINT2D_DTYPE.itemsize

INVALID_POINT3D_ID = 0xFFFF_FFFF_FFFF_FFFF
MODEL_FILENAMES = ("cameras.bin", "images.bin", "points3D.bin")
_MAX_NAME_BYTES = 4096


@dataclass(frozen=True)
class CameraModelSpec:
    model_id: int
    name: str
    param_names: tuple[str, ...]

    @property
    def num_params(self) -> int:
        return len(self.param_names)


def _spec(model_id: int, name: str, *params: str) -> CameraModelSpec:
    return CameraModelSpec(model_id, name, params)


# Confirmed against the parameter-name strings in COLMAP 3.9.1. Ids 0..10 only;
# RAD_TAN_THIN_PRISM_FISHEYE (id 11) arrived in COLMAP 3.10.
CAMERA_MODELS: dict[int, CameraModelSpec] = {
    m.model_id: m
    for m in (
        _spec(0, "SIMPLE_PINHOLE", "f", "cx", "cy"),
        _spec(1, "PINHOLE", "fx", "fy", "cx", "cy"),
        _spec(2, "SIMPLE_RADIAL", "f", "cx", "cy", "k"),
        _spec(3, "RADIAL", "f", "cx", "cy", "k1", "k2"),
        _spec(4, "OPENCV", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
        _spec(5, "OPENCV_FISHEYE", "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"),
        _spec(
            6, "FULL_OPENCV",
            "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
        ),
        _spec(7, "FOV", "fx", "fy", "cx", "cy", "omega"),
        _spec(8, "SIMPLE_RADIAL_FISHEYE", "f", "cx", "cy", "k"),
        _spec(9, "RADIAL_FISHEYE", "f", "cx", "cy", "k1", "k2"),
        _spec(
            10, "THIN_PRISM_FISHEYE",
            "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1",
        ),
    )
}

CAMERA_MODELS_BY_NAME: dict[str, CameraModelSpec] = {
    spec.name: spec for spec in CAMERA_MODELS.values()
}


class _Cursor:
    """Bounds-checked read cursor over an in-memory model file.

    Centralising the bounds checks means a short read is reported with its file
    and byte offset from one place, instead of being spread over a dozen
    call sites where one could be forgotten.
    """

    def __init__(self, data: bytes, path: Path) -> None:
        self._data = data
        self._path = path
        self.offset = 0

    @property
    def remaining(self) -> int:
        return len(self._data) - self.offset

    def _fail(self, what: str, need: int) -> None:
        raise TruncatedFileError(
            f"{self._path}: truncated at byte {self.offset} reading {what}: "
            f"need {need} bytes, {self.remaining} remain "
            f"(file is {len(self._data)} bytes)"
        )

    def unpack(self, fmt: struct.Struct, what: str) -> tuple:
        if self.remaining < fmt.size:
            self._fail(what, fmt.size)
        values = fmt.unpack_from(self._data, self.offset)
        self.offset += fmt.size
        return values

    def read_u64(self, what: str) -> int:
        return self.unpack(_U64, what)[0]

    def read_cstring(self, what: str) -> str:
        end = self._data.find(b"\x00", self.offset)
        if end == -1:
            self._fail(what, self.remaining + 1)
        if end - self.offset > _MAX_NAME_BYTES:
            raise TruncatedFileError(
                f"{self._path}: {what} at byte {self.offset} exceeds "
                f"{_MAX_NAME_BYTES} bytes without a NUL terminator"
            )
        raw = self._data[self.offset : end]
        self.offset = end + 1
        # surrogateescape round-trips non-UTF-8 filenames back to the filesystem
        # intact; 'replace' would silently corrupt them and 'strict' would
        # reject legitimate names.
        return raw.decode("utf-8", errors="surrogateescape")

    def read_array(self, dtype: np.dtype, count: int, what: str) -> np.ndarray:
        nbytes = dtype.itemsize * count
        if self.remaining < nbytes:
            self._fail(what, nbytes)
        array = np.frombuffer(self._data, dtype=dtype, count=count, offset=self.offset)
        self.offset += nbytes
        return array

    def skip(self, nbytes: int, what: str) -> None:
        if self.remaining < nbytes:
            self._fail(what, nbytes)
        self.offset += nbytes

    def require_capacity(self, count: int, min_record: int, what: str) -> None:
        """Reject an implausible record count before allocating anything."""
        if count < 0 or count * min_record > self.remaining:
            raise TruncatedFileError(
                f"{self._path}: declares {count} {what} needing at least "
                f"{count * min_record} bytes, but only {self.remaining} remain"
            )

    def require_eof(self, what: str) -> None:
        """Trailing bytes mean we misread something; landing exactly on EOF is
        strong evidence every field width above was right."""
        if self.remaining:
            raise ColmapModelError(
                f"{self._path}: {self.remaining} unexpected trailing byte(s) after "
                f"reading {what}; the file layout does not match what this reader expects"
            )


def qvec_to_rotmat(qvec: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    """Rotation matrix for a COLMAP (w, x, y, z) quaternion (world -> camera)."""
    q = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm == 0.0:
        raise InconsistentModelError(f"quaternion is not a valid rotation: {tuple(q)}")
    if abs(norm - 1.0) > 1e-6:
        # A non-unit quaternion yields a matrix that is not a rotation, which
        # would skew every camera centre without any obvious symptom.
        logger.warning("Quaternion norm %.9f differs from 1; normalizing", norm)
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model_id: int
    width: int
    height: int
    params: tuple[float, ...]

    @property
    def model(self) -> CameraModelSpec:
        return CAMERA_MODELS[self.model_id]

    @property
    def model_name(self) -> str:
        return self.model.name

    def param(self, name: str) -> float:
        spec = self.model
        try:
            return self.params[spec.param_names.index(name)]
        except ValueError:
            raise KeyError(f"{spec.name} has no parameter {name!r}") from None

    @property
    def focal_xy(self) -> tuple[float, float]:
        names = self.model.param_names
        if "f" in names:
            f = self.param("f")
            return f, f
        return self.param("fx"), self.param("fy")

    @property
    def principal_point(self) -> tuple[float, float]:
        return self.param("cx"), self.param("cy")


@dataclass(eq=False)
class Image:
    """A registered image. `eq=False` because the ndarray fields make the
    generated __eq__ raise on ambiguous truth value."""

    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    num_points2d: int
    points2d: np.ndarray | None = None

    @property
    def xys(self) -> np.ndarray:
        if self.points2d is None:
            raise ValueError(f"image {self.name}: point2D data was not read")
        return self.points2d["xy"]

    @property
    def point3d_ids(self) -> np.ndarray:
        if self.points2d is None:
            raise ValueError(f"image {self.name}: point2D data was not read")
        return self.points2d["point3D_id"]

    @property
    def num_valid_points3d(self) -> int:
        return int(np.count_nonzero(self.point3d_ids != INVALID_POINT3D_ID))

    def valid_point3d_ids(self) -> np.ndarray:
        ids = self.point3d_ids
        return ids[ids != INVALID_POINT3D_ID]

    def rotation_matrix(self) -> np.ndarray:
        return qvec_to_rotmat(self.qvec)

    def projection_center(self) -> np.ndarray:
        """Camera centre in world coordinates: C = -R^T t."""
        return -self.rotation_matrix().T @ np.asarray(self.tvec, dtype=np.float64)

    def viewing_direction(self) -> np.ndarray:
        """Unit vector along the camera's optical axis, in world coordinates."""
        return self.rotation_matrix().T @ np.array([0.0, 0.0, 1.0])

    @property
    def group(self) -> str:
        """Leading path component of the image name.

        `organize` lays images out as `<group>/<file>`, so this is the camera
        model / resolution bucket the image came from.
        """
        return self.name.split("/")[0] if "/" in self.name else ""


@dataclass(eq=False)
class Point3D:
    """A single 3D point, materialised on demand as a view into `Points3D`."""

    point3d_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: np.ndarray  # (M, 2) uint32: [image_id, point2D_idx]

    @property
    def track_length(self) -> int:
        return int(self.track.shape[0])

    @property
    def track_image_ids(self) -> np.ndarray:
        return self.track[:, 0]


@dataclass(eq=False)
class Points3D:
    """Columnar point cloud with CSR-style variable-length tracks.

    Half a million small objects would cost several hundred MB and force every
    statistic through a hand-written loop. Columnar arrays keep it to tens of
    MB and let each statistic be a single vectorised expression, which is
    harder to get subtly wrong.
    """

    ids: np.ndarray  # (P,) uint64
    xyz: np.ndarray  # (P, 3) float64
    rgb: np.ndarray  # (P, 3) uint8
    error: np.ndarray  # (P,) float64
    track_offsets: np.ndarray  # (P+1,) int64
    track_image_ids: np.ndarray  # (T,) uint32
    track_point2d_idxs: np.ndarray  # (T,) uint32
    _index: dict[int, int] | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    @property
    def num_observations(self) -> int:
        return int(self.track_image_ids.shape[0])

    def track_lengths(self) -> np.ndarray:
        return np.diff(self.track_offsets)

    def track_of_index(self, i: int) -> np.ndarray:
        lo, hi = int(self.track_offsets[i]), int(self.track_offsets[i + 1])
        return np.column_stack((self.track_image_ids[lo:hi], self.track_point2d_idxs[lo:hi]))

    def index_of(self, point3d_id: int) -> int:
        if self._index is None:
            self._index = {int(pid): i for i, pid in enumerate(self.ids)}
        return self._index[point3d_id]

    def point_at(self, i: int) -> Point3D:
        return Point3D(
            point3d_id=int(self.ids[i]),
            xyz=self.xyz[i],
            rgb=self.rgb[i],
            error=float(self.error[i]),
            track=self.track_of_index(i),
        )

    def __getitem__(self, point3d_id: int) -> Point3D:
        return self.point_at(self.index_of(point3d_id))

    def __iter__(self) -> Iterator[Point3D]:
        for i in range(len(self)):
            yield self.point_at(i)


@dataclass(eq=False)
class Model:
    path: Path
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points3d: Points3D

    @property
    def num_registered_images(self) -> int:
        return len(self.images)

    @property
    def num_points3d(self) -> int:
        return len(self.points3d)

    @property
    def num_observations(self) -> int:
        return self.points3d.num_observations

    def images_by_name(self) -> dict[str, Image]:
        by_name: dict[str, Image] = {}
        for image in self.images.values():
            if image.name in by_name:
                raise InconsistentModelError(
                    f"{self.path}: duplicate image name {image.name!r} "
                    f"(image ids {by_name[image.name].image_id} and {image.image_id})"
                )
            by_name[image.name] = image
        return by_name

    def mean_track_length(self) -> float:
        if not len(self.points3d):
            return 0.0
        return float(self.points3d.num_observations / len(self.points3d))

    def mean_observations_per_image(self) -> float:
        if not self.images:
            return 0.0
        return float(self.points3d.num_observations / len(self.images))

    def mean_reprojection_error(self) -> float:
        if not len(self.points3d):
            return 0.0
        return float(np.mean(self.points3d.error))

    def observations_per_image(self) -> dict[int, int]:
        """Number of 3D-point observations contributed by each image."""
        counts = dict.fromkeys(self.images, 0)
        if self.points3d.num_observations:
            ids, tallies = np.unique(self.points3d.track_image_ids, return_counts=True)
            for image_id, tally in zip(ids.tolist(), tallies.tolist(), strict=True):
                counts[image_id] = tally
        return counts

    def camera_centers(self) -> tuple[list[int], np.ndarray]:
        image_ids = sorted(self.images)
        if not image_ids:
            return [], np.zeros((0, 3), dtype=np.float64)
        centers = np.array(
            [self.images[i].projection_center() for i in image_ids], dtype=np.float64
        )
        return image_ids, centers

    def viewing_directions(self, image_ids: list[int]) -> np.ndarray:
        if not image_ids:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array(
            [self.images[i].viewing_direction() for i in image_ids], dtype=np.float64
        )


def _read_file(path: Path) -> _Cursor:
    if not path.is_file():
        raise MissingModelFileError(f"Model file not found: {path}")
    return _Cursor(path.read_bytes(), path)


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cursor = _read_file(path)
    num_cameras = cursor.read_u64("camera count")
    # Smallest possible record: header + the 3-param SIMPLE_PINHOLE model.
    cursor.require_capacity(num_cameras, _CAMERA_HEADER.size + 24, "cameras")

    cameras: dict[int, Camera] = {}
    for index in range(num_cameras):
        camera_id, model_id, width, height = cursor.unpack(
            _CAMERA_HEADER, f"camera {index} header"
        )
        spec = CAMERA_MODELS.get(model_id)
        if spec is None:
            # Guessing a parameter count would desynchronise the whole file.
            raise UnknownCameraModelError(
                f"{path}: camera {camera_id} uses unknown camera model id {model_id}; "
                f"known ids are {sorted(CAMERA_MODELS)} "
                f"(a newer COLMAP may have added this model)"
            )
        params = cursor.unpack(
            struct.Struct(f"<{spec.num_params}d"), f"camera {camera_id} params"
        )
        if not all(np.isfinite(params)):
            raise InconsistentModelError(f"{path}: camera {camera_id} has non-finite params")
        if camera_id in cameras:
            raise InconsistentModelError(f"{path}: duplicate camera id {camera_id}")
        cameras[camera_id] = Camera(camera_id, model_id, width, height, params)

    cursor.require_eof("cameras")
    return cameras


def read_images_binary(path: Path, *, read_points2d: bool = True) -> dict[int, Image]:
    cursor = _read_file(path)
    num_images = cursor.read_u64("image count")
    # Header + shortest possible name ("x\0") + the point2D count.
    cursor.require_capacity(num_images, _IMAGE_HEADER.size + 2 + 8, "images")

    images: dict[int, Image] = {}
    for index in range(num_images):
        values = cursor.unpack(_IMAGE_HEADER, f"image {index} header")
        image_id = values[0]
        qvec = values[1:5]
        tvec = values[5:8]
        camera_id = values[8]

        name = cursor.read_cstring(f"image {image_id} name")
        if not name:
            raise InconsistentModelError(f"{path}: image {image_id} has an empty name")
        if not all(np.isfinite(qvec)) or not all(np.isfinite(tvec)):
            raise InconsistentModelError(f"{path}: image {image_id} has a non-finite pose")

        num_points2d = cursor.read_u64(f"image {image_id} point2D count")
        cursor.require_capacity(
            num_points2d, _POINT2D_DTYPE.itemsize, f"point2D records for image {image_id}"
        )
        if read_points2d:
            points2d = cursor.read_array(
                _POINT2D_DTYPE, num_points2d, f"image {image_id} point2D block"
            )
        else:
            points2d = None
            cursor.skip(
                _POINT2D_DTYPE.itemsize * num_points2d, f"image {image_id} point2D block"
            )

        if image_id in images:
            raise InconsistentModelError(f"{path}: duplicate image id {image_id}")
        images[image_id] = Image(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            num_points2d=num_points2d,
            points2d=points2d,
        )

    cursor.require_eof("images")
    return images


def read_points3d_binary(path: Path) -> Points3D:
    cursor = _read_file(path)
    num_points = cursor.read_u64("point3D count")
    cursor.require_capacity(num_points, _POINT3D_HEADER.size, "3D points")

    ids = np.empty(num_points, dtype=np.uint64)
    xyz = np.empty((num_points, 3), dtype=np.float64)
    rgb = np.empty((num_points, 3), dtype=np.uint8)
    error = np.empty(num_points, dtype=np.float64)
    track_lengths = np.empty(num_points, dtype=np.int64)
    track_chunks: list[bytes] = []
    seen: set[int] = set()

    # Tracks are variable-length, so there is no correct way to bulk-read this
    # file in one call -- assuming a uniform track length would be silently
    # wrong on every real model.
    for index in range(num_points):
        values = cursor.unpack(_POINT3D_HEADER, f"point3D {index} header")
        point_id = values[0]
        if point_id in seen:
            raise InconsistentModelError(f"{path}: duplicate point3D id {point_id}")
        seen.add(point_id)

        ids[index] = point_id
        xyz[index] = values[1:4]
        rgb[index] = values[4:7]
        error[index] = values[7]
        track_length = values[8]
        track_lengths[index] = track_length

        nbytes = 8 * track_length
        if cursor.remaining < nbytes:
            cursor._fail(f"point3D {point_id} track", nbytes)
        track_chunks.append(cursor._data[cursor.offset : cursor.offset + nbytes])
        cursor.offset += nbytes

    cursor.require_eof("3D points")

    if track_chunks:
        flat = np.frombuffer(b"".join(track_chunks), dtype="<u4").reshape(-1, 2)
        track_image_ids = np.ascontiguousarray(flat[:, 0])
        track_point2d_idxs = np.ascontiguousarray(flat[:, 1])
    else:
        track_image_ids = np.zeros(0, dtype=np.uint32)
        track_point2d_idxs = np.zeros(0, dtype=np.uint32)

    track_offsets = np.zeros(num_points + 1, dtype=np.int64)
    np.cumsum(track_lengths, out=track_offsets[1:])

    if not np.all(np.isfinite(xyz)):
        raise InconsistentModelError(f"{path}: model contains non-finite 3D point coordinates")

    return Points3D(
        ids=ids,
        xyz=xyz,
        rgb=rgb,
        error=error,
        track_offsets=track_offsets,
        track_image_ids=track_image_ids,
        track_point2d_idxs=track_point2d_idxs,
    )


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    count: int = 1


def _summarize(offenders: list[str], total: int) -> str:
    shown = ", ".join(offenders[:5])
    suffix = f" (first 5 of {total})" if total > 5 else ""
    return f"{shown}{suffix}"


def validate_model(model: Model) -> list[ValidationIssue]:
    """Cross-file consistency checks between cameras, images and points3D."""
    issues: list[ValidationIssue] = []

    missing_cameras = [
        f"image {img.image_id} -> camera {img.camera_id}"
        for img in model.images.values()
        if img.camera_id not in model.cameras
    ]
    if missing_cameras:
        issues.append(
            ValidationIssue(
                "error",
                "unknown_camera_id",
                f"{len(missing_cameras)} image(s) reference a camera absent from "
                f"cameras.bin: {_summarize(missing_cameras, len(missing_cameras))}",
                len(missing_cameras),
            )
        )

    points = model.points3d
    if points.num_observations:
        known_ids = np.fromiter(model.images, dtype=np.int64, count=len(model.images))
        track_ids = points.track_image_ids.astype(np.int64)
        unknown_mask = ~np.isin(track_ids, known_ids)
        if unknown_mask.any():
            bad = np.unique(track_ids[unknown_mask]).tolist()
            issues.append(
                ValidationIssue(
                    "error",
                    "track_unknown_image",
                    f"{int(unknown_mask.sum())} track element(s) reference image ids "
                    f"absent from images.bin: {_summarize([str(b) for b in bad], len(bad))}",
                    int(unknown_mask.sum()),
                )
            )
        else:
            # Bidirectional check: for every track entry (image, idx) of point P,
            # that image's point3D_ids[idx] must be P. This independently
            # cross-validates the images.bin and points3D.bin parsers against
            # each other -- a misparse of either will almost certainly break it.
            out_of_range = 0
            mismatched = 0
            examples: list[str] = []
            have_points2d = all(img.points2d is not None for img in model.images.values())
            expanded = np.repeat(points.ids, points.track_lengths())
            for image_id, idx, expected in zip(
                points.track_image_ids.tolist(),
                points.track_point2d_idxs.tolist(),
                expanded.tolist(),
                strict=True,
            ):
                image = model.images[image_id]
                if idx >= image.num_points2d:
                    out_of_range += 1
                    if len(examples) < 5:
                        examples.append(
                            f"point {expected} -> image {image_id} index {idx} "
                            f"(image has {image.num_points2d})"
                        )
                elif have_points2d and int(image.point3d_ids[idx]) != expected:
                    mismatched += 1
                    if len(examples) < 5:
                        examples.append(
                            f"point {expected} -> image {image_id} index {idx} "
                            f"which points back at {int(image.point3d_ids[idx])}"
                        )
            if out_of_range:
                issues.append(
                    ValidationIssue(
                        "error",
                        "track_index_out_of_range",
                        f"{out_of_range} track element(s) index past the end of their "
                        f"image's point2D list: {_summarize(examples, out_of_range)}",
                        out_of_range,
                    )
                )
            if mismatched:
                issues.append(
                    ValidationIssue(
                        "error",
                        "track_point2d_mismatch",
                        f"{mismatched} track element(s) disagree with the image's own "
                        f"point3D id: {_summarize(examples, mismatched)}",
                        mismatched,
                    )
                )

    lengths = points.track_lengths()
    degenerate = int(np.count_nonzero(lengths < 2))
    if degenerate:
        issues.append(
            ValidationIssue(
                "warning",
                "degenerate_track",
                f"{degenerate} 3D point(s) have a track shorter than 2 observations",
                degenerate,
            )
        )

    for camera in model.cameras.values():
        fx, fy = camera.focal_xy
        if fx <= 0 or fy <= 0 or fx > 10 * camera.width:
            issues.append(
                ValidationIssue(
                    "warning",
                    "implausible_intrinsics",
                    f"camera {camera.camera_id} ({camera.model_name}) has implausible "
                    f"focal length ({fx:.2f}, {fy:.2f}) for a {camera.width}x"
                    f"{camera.height} image",
                )
            )

    return issues


def read_model(
    model_dir: Path, *, read_points2d: bool = True, validate: bool = True
) -> Model:
    """Read a full COLMAP sparse model from a directory."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise MissingModelFileError(f"Model directory not found: {model_dir}")

    model = Model(
        path=model_dir,
        cameras=read_cameras_binary(model_dir / "cameras.bin"),
        images=read_images_binary(model_dir / "images.bin", read_points2d=read_points2d),
        points3d=read_points3d_binary(model_dir / "points3D.bin"),
    )

    if validate:
        issues = validate_model(model)
        for issue in issues:
            if issue.severity == "warning":
                logger.warning("%s: %s", model_dir, issue.message)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            detail = "; ".join(i.message for i in errors)
            raise InconsistentModelError(f"{model_dir}: {detail}")

    return model


def _is_model_dir(path: Path) -> bool:
    return all((path / name).is_file() for name in MODEL_FILENAMES)


def _missing_files(path: Path) -> list[str]:
    return [name for name in MODEL_FILENAMES if not (path / name).is_file()]


def find_models(sparse_dir: Path) -> list[Path]:
    """Find every COLMAP sparse model directory under `sparse_dir`.

    Handles both the numbered-submodel layout (`sparse/0`, `sparse/1`, ...) and
    a bare `sparse/` holding the three files directly. A directory holding only
    *some* of the files raises rather than being skipped: a half-written
    `sparse/1/` is exactly the case where skipping would let a caller conclude
    "one model, looks fine" after a crashed run.
    """
    sparse_dir = Path(sparse_dir)
    if not sparse_dir.is_dir():
        raise MissingModelFileError(f"Sparse model directory not found: {sparse_dir}")

    if _is_model_dir(sparse_dir):
        return [sparse_dir]

    models: list[Path] = []
    for child in sorted(p for p in sparse_dir.iterdir() if p.is_dir()):
        missing = _missing_files(child)
        if not missing:
            models.append(child)
        elif len(missing) < len(MODEL_FILENAMES):
            raise MissingModelFileError(
                f"{child} looks like a partially written model: missing "
                f"{', '.join(missing)}. This usually means the mapper crashed "
                f"or is still running."
            )

    if not models:
        present = sorted(p.name for p in sparse_dir.iterdir())
        hint = ""
        if any(name.endswith(".txt") for name in present):
            hint = (
                " Found .txt model files; convert them with "
                "`colmap model_converter --output_type BIN`."
            )
        raise MissingModelFileError(
            f"No COLMAP model found in {sparse_dir} (contains: "
            f"{', '.join(present) if present else 'nothing'}).{hint}"
        )

    # Numeric sort so sparse/10 follows sparse/9 rather than sparse/1.
    if all(p.name.isdigit() for p in models):
        models.sort(key=lambda p: int(p.name))
    return models


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_cameras_binary(path: Path, cameras: dict[int, Camera]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [_U64.pack(len(cameras))]
    for camera_id in sorted(cameras):
        camera = cameras[camera_id]
        spec = CAMERA_MODELS.get(camera.model_id)
        if spec is None:
            raise UnknownCameraModelError(
                f"cannot write camera {camera_id}: unknown model id {camera.model_id}"
            )
        if len(camera.params) != spec.num_params:
            raise InconsistentModelError(
                f"camera {camera_id} ({spec.name}) has {len(camera.params)} params, "
                f"expected {spec.num_params}"
            )
        chunks.append(
            _CAMERA_HEADER.pack(camera_id, camera.model_id, camera.width, camera.height)
        )
        chunks.append(struct.pack(f"<{spec.num_params}d", *camera.params))
    path.write_bytes(b"".join(chunks))


def write_images_binary(path: Path, images: dict[int, Image]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [_U64.pack(len(images))]
    for image_id in sorted(images):
        image = images[image_id]
        chunks.append(
            _IMAGE_HEADER.pack(image_id, *image.qvec, *image.tvec, image.camera_id)
        )
        chunks.append(image.name.encode("utf-8", errors="surrogateescape") + b"\x00")
        if image.points2d is None:
            # Reading with read_points2d=False discards the feature table; a
            # model written from that would silently lose every 2D observation,
            # so refuse rather than emit a lossy file.
            raise InconsistentModelError(
                f"image {image.name} was read without its point2D data and cannot be "
                f"written; re-read the model with read_points2d=True"
            )
        chunks.append(_U64.pack(len(image.points2d)))
        chunks.append(image.points2d.astype(_POINT2D_DTYPE, copy=False).tobytes())
    path.write_bytes(b"".join(chunks))


def write_points3d_binary(path: Path, points: Points3D) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [_U64.pack(len(points))]
    offsets = points.track_offsets
    for i in range(len(points)):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        chunks.append(
            _POINT3D_HEADER.pack(
                int(points.ids[i]),
                float(points.xyz[i, 0]), float(points.xyz[i, 1]), float(points.xyz[i, 2]),
                int(points.rgb[i, 0]), int(points.rgb[i, 1]), int(points.rgb[i, 2]),
                float(points.error[i]),
                hi - lo,
            )
        )
        if hi > lo:
            track = np.empty((hi - lo, 2), dtype="<u4")
            track[:, 0] = points.track_image_ids[lo:hi]
            track[:, 1] = points.track_point2d_idxs[lo:hi]
            chunks.append(track.tobytes())
    path.write_bytes(b"".join(chunks))


def write_model(model_dir: Path, model: Model) -> Path:
    """Write a model as the three COLMAP binaries. Returns the directory."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    write_cameras_binary(model_dir / "cameras.bin", model.cameras)
    write_images_binary(model_dir / "images.bin", model.images)
    write_points3d_binary(model_dir / "points3D.bin", model.points3d)
    return model_dir


@dataclass
class SubsetStats:
    images: int
    cameras: int
    points: int
    observations: int
    points_dropped_short_track: int


def subset_model(
    model: Model,
    image_ids: set[int],
    point_ids: set[int],
    *,
    min_track_length: int = 2,
    path: Path | None = None,
) -> tuple[Model, SubsetStats]:
    """Build a self-consistent sub-model from a set of images and 3D points.

    The two sides have to be reconciled or the result is not a valid model:
    kept points may be observed by dropped images, and kept images certainly
    observe dropped points. So tracks are filtered to kept images, and each
    kept image's feature table has its references to dropped points reset to
    the invalid sentinel.

    Each image keeps its *full* 2D feature list rather than a compacted one.
    That is deliberate: track entries address features by index, so renumbering
    them would invalidate every `point2D_idx` in the model.
    """
    kept_images = {i: model.images[i] for i in sorted(image_ids) if i in model.images}
    if not kept_images:
        raise InconsistentModelError("subset would contain no images")

    points = model.points3d
    keep_indices: list[int] = []
    new_tracks: list[np.ndarray] = []
    dropped_short = 0

    for i in range(len(points)):
        if int(points.ids[i]) not in point_ids:
            continue
        lo, hi = int(points.track_offsets[i]), int(points.track_offsets[i + 1])
        image_slice = points.track_image_ids[lo:hi]
        mask = np.fromiter(
            (int(v) in kept_images for v in image_slice.tolist()), dtype=bool, count=hi - lo
        )
        if int(mask.sum()) < min_track_length:
            dropped_short += 1
            continue
        keep_indices.append(i)
        track = np.empty((int(mask.sum()), 2), dtype=np.uint32)
        track[:, 0] = image_slice[mask]
        track[:, 1] = points.track_point2d_idxs[lo:hi][mask]
        new_tracks.append(track)

    if keep_indices:
        index = np.array(keep_indices, dtype=np.int64)
        lengths = np.array([t.shape[0] for t in new_tracks], dtype=np.int64)
        flat = np.concatenate(new_tracks) if new_tracks else np.zeros((0, 2), dtype=np.uint32)
        track_offsets = np.zeros(len(keep_indices) + 1, dtype=np.int64)
        np.cumsum(lengths, out=track_offsets[1:])
        new_points = Points3D(
            ids=points.ids[index].copy(),
            xyz=points.xyz[index].copy(),
            rgb=points.rgb[index].copy(),
            error=points.error[index].copy(),
            track_offsets=track_offsets,
            track_image_ids=np.ascontiguousarray(flat[:, 0]),
            track_point2d_idxs=np.ascontiguousarray(flat[:, 1]),
        )
    else:
        new_points = Points3D(
            ids=np.zeros(0, dtype=np.uint64),
            xyz=np.zeros((0, 3), dtype=np.float64),
            rgb=np.zeros((0, 3), dtype=np.uint8),
            error=np.zeros(0, dtype=np.float64),
            track_offsets=np.zeros(1, dtype=np.int64),
            track_image_ids=np.zeros(0, dtype=np.uint32),
            track_point2d_idxs=np.zeros(0, dtype=np.uint32),
        )

    surviving = set(new_points.ids.tolist())
    rebuilt: dict[int, Image] = {}
    for image_id, image in kept_images.items():
        if image.points2d is None:
            raise InconsistentModelError(
                f"image {image.name} was read without point2D data; re-read the model "
                f"with read_points2d=True before subsetting"
            )
        features = image.points2d.copy()
        ids = features["point3D_id"]
        drop = ids != INVALID_POINT3D_ID
        if drop.any():
            keep_mask = np.isin(ids, np.fromiter(surviving, dtype=np.uint64, count=len(surviving)))
            ids[~keep_mask] = INVALID_POINT3D_ID
        rebuilt[image_id] = Image(
            image_id=image.image_id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            num_points2d=image.num_points2d,
            points2d=features,
        )

    camera_ids = {img.camera_id for img in rebuilt.values()}
    missing = camera_ids - set(model.cameras)
    if missing:
        raise InconsistentModelError(f"subset references unknown camera id(s) {sorted(missing)}")
    cameras = {cid: model.cameras[cid] for cid in sorted(camera_ids)}

    subset = Model(
        path=path or model.path,
        cameras=cameras,
        images=rebuilt,
        points3d=new_points,
    )
    stats = SubsetStats(
        images=len(rebuilt),
        cameras=len(cameras),
        points=len(new_points),
        observations=new_points.num_observations,
        points_dropped_short_track=dropped_short,
    )
    return subset, stats
