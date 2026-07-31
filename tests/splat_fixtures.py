"""Fixtures for chunk/merge: a synthetic multi-room scene and 3DGS PLY files.

The room builder produces the geometry the `chunk` selection rules actually
care about: cameras standing inside each room, points belonging to each room,
and -- critically -- a camera in one room angled through a doorway so that it
observes a meaningful number of the *neighbouring* room's points. That
through-doorway observer is the case that a naive "camera centre inside the
box" rule silently drops.

Like `colmap_fixtures`, the PLY writer here builds its bytes by hand rather
than calling `pipeline.ply`, so a round-trip test compares two independent
implementations instead of one implementation with itself.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .colmap_fixtures import (
    FixtureCamera,
    FixtureImage,
    FixtureModel,
    FixturePoint3D,
    make_consistent,
)

# A 3DGS splat at spherical-harmonic degree 3: 3 DC + 45 rest coefficients.
SH_REST_COUNT = 45


def gaussian_property_names(sh_rest: int = SH_REST_COUNT) -> list[str]:
    """The property list a real 3DGS PLY carries, in the usual order."""
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(sh_rest)]
    names += ["opacity", "scale_0", "scale_1", "scale_2"]
    names += ["rot_0", "rot_1", "rot_2", "rot_3"]
    return names


def write_gaussian_ply(
    path: Path,
    xyz: np.ndarray,
    *,
    sh_rest: int = SH_REST_COUNT,
    fmt: str = "binary_little_endian",
    seed: int = 0,
    extra_comment: str | None = None,
) -> dict[str, np.ndarray]:
    """Write a 3DGS PLY at the given positions with deterministic other fields.

    Returns the full field table so tests can assert every property survived a
    read/crop/merge round trip, not just the positions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float64)
    count = xyz.shape[0]
    rng = np.random.default_rng(seed)
    names = gaussian_property_names(sh_rest)

    fields: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        if name == "x":
            fields[name] = xyz[:, 0].astype(np.float32)
        elif name == "y":
            fields[name] = xyz[:, 1].astype(np.float32)
        elif name == "z":
            fields[name] = xyz[:, 2].astype(np.float32)
        else:
            # Distinct per-property values, so a test can detect a column that
            # got dropped, duplicated or reordered.
            fields[name] = (
                rng.random(count).astype(np.float32) + np.float32(i)
            ).astype(np.float32)

    header = ["ply", f"format {fmt} 1.0"]
    if extra_comment:
        header.append(f"comment {extra_comment}")
    header.append(f"element vertex {count}")
    header += [f"property float {name}" for name in names]
    header.append("end_header")

    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        if fmt == "ascii":
            for row in range(count):
                handle.write(
                    (" ".join(repr(float(fields[n][row])) for n in names) + "\n").encode("ascii")
                )
        else:
            order = "<" if fmt == "binary_little_endian" else ">"
            for row in range(count):
                handle.write(
                    struct.pack(f"{order}{len(names)}f", *(float(fields[n][row]) for n in names))
                )

    return fields


@dataclass
class Room:
    name: str
    box_min: tuple[float, float, float]
    box_max: tuple[float, float, float]
    image_ids: list[int]
    point_ids: list[int]


@dataclass
class SceneFixture:
    model: FixtureModel
    rooms: list[Room]
    doorway_image_id: int
    doorway_sees_room: str
    doorway_shared_points: int


def build_two_room_scene(
    *,
    cameras_per_room: int = 6,
    points_per_room: int = 40,
    doorway_observations: int = 12,
    room_a_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    room_b_origin: tuple[float, float, float] = (20.0, 0.0, 0.0),
    room_size: float = 8.0,
    seed: int = 0,
) -> SceneFixture:
    """Two rooms plus one camera in room A that sees into room B.

    Layout, along x:

        room A: cameras and points around `room_a_origin`
        room B: cameras and points around `room_b_origin`
        one extra camera, positioned inside room A, whose track includes
        `doorway_observations` of room B's points

    The doorway camera's centre is inside A's box only, so any selection rule
    based purely on camera position will leave it out of B -- which is exactly
    what the through-doorway rule exists to prevent.
    """
    rng = np.random.default_rng(seed)
    features_per_image = max(points_per_room, doorway_observations) + 10

    cameras = [FixtureCamera(camera_id=1, model="PINHOLE", width=1920, height=1080)]
    images: list[FixtureImage] = []
    points: list[FixturePoint3D] = []
    rooms: list[Room] = []

    next_image_id = 1
    next_point_id = 1

    for room_index, origin in enumerate((room_a_origin, room_b_origin)):
        name = "room_a" if room_index == 0 else "room_b"
        base = np.asarray(origin, dtype=np.float64)
        half = room_size / 2.0

        room_image_ids: list[int] = []
        for c in range(cameras_per_room):
            # Cameras sit in a small cluster well inside the room's box.
            offset = np.array([
                (c / max(cameras_per_room - 1, 1) - 0.5) * room_size * 0.5,
                0.0,
                (c % 2) * 0.5,
            ])
            center = base + offset
            # Identity rotation, so tvec = -center puts the camera at `center`.
            images.append(
                FixtureImage(
                    image_id=next_image_id,
                    name=f"{name}/frame_{c:06d}.jpg",
                    camera_id=1,
                    qvec=(1.0, 0.0, 0.0, 0.0),
                    tvec=tuple((-center).tolist()),
                    points2d=[
                        (float(rng.integers(0, 1920)), float(rng.integers(0, 1080)), None)
                        for _ in range(features_per_image)
                    ],
                )
            )
            room_image_ids.append(next_image_id)
            next_image_id += 1

        room_point_ids: list[int] = []
        slot = dict.fromkeys(room_image_ids, 0)
        for p in range(points_per_room):
            position = base + np.array([
                (rng.random() - 0.5) * room_size * 0.9,
                (rng.random() - 0.5) * room_size * 0.9,
                (rng.random() - 0.5) * room_size * 0.9,
            ])
            # Each point is seen by three of the room's cameras.
            track = []
            for k in range(3):
                image_id = room_image_ids[(p + k) % len(room_image_ids)]
                track.append((image_id, slot[image_id]))
                slot[image_id] += 1
            points.append(
                FixturePoint3D(
                    point3d_id=next_point_id,
                    xyz=tuple(position.tolist()),
                    rgb=(120, 120, 120),
                    error=0.5,
                    track=track,
                )
            )
            room_point_ids.append(next_point_id)
            next_point_id += 1

        rooms.append(
            Room(
                name=name,
                box_min=tuple((base - half).tolist()),
                box_max=tuple((base + half).tolist()),
                image_ids=room_image_ids,
                point_ids=room_point_ids,
            )
        )

    # The doorway camera: standing in room A, observing room B's points.
    doorway_id = next_image_id
    doorway_center = np.asarray(room_a_origin, dtype=np.float64) + np.array([1.0, 0.0, 0.0])
    images.append(
        FixtureImage(
            image_id=doorway_id,
            name="room_a/doorway_000000.jpg",
            camera_id=1,
            qvec=(1.0, 0.0, 0.0, 0.0),
            tvec=tuple((-doorway_center).tolist()),
            points2d=[
                (float(rng.integers(0, 1920)), float(rng.integers(0, 1080)), None)
                for _ in range(features_per_image)
            ],
        )
    )

    room_b_points = {p.point3d_id for p in points if p.point3d_id in set(rooms[1].point_ids)}
    targets = sorted(room_b_points)[:doorway_observations]
    for slot_index, point_id in enumerate(targets):
        point = next(p for p in points if p.point3d_id == point_id)
        point.track.append((doorway_id, slot_index))

    model = make_consistent(FixtureModel(cameras=cameras, images=images, points3d=points))
    return SceneFixture(
        model=model,
        rooms=rooms,
        doorway_image_id=doorway_id,
        doorway_sees_room="room_b",
        doorway_shared_points=len(targets),
    )
