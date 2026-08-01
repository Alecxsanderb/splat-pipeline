"""Named axis-aligned bounding boxes, and the YAML they are declared in.

`chunk` carves a reconstruction into one box per room; `merge` crops each
trained chunk back to its box so overlapping chunks do not leave doubled
geometry at the walls. Both ends read the same box definition, so it lives
here rather than in either command.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class BoxError(ValueError):
    """Raised when a box definition or box file is malformed."""


@dataclass(frozen=True)
class BoundingBox:
    name: str
    min: tuple[float, float, float]
    max: tuple[float, float, float]

    def __post_init__(self) -> None:
        for lo, hi, axis in zip(self.min, self.max, "xyz", strict=True):
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise BoxError(f"box {self.name!r} has a non-finite {axis} bound")
            if hi < lo:
                raise BoxError(
                    f"box {self.name!r} has max {axis}={hi} below min {axis}={lo}; "
                    f"min must be the lower corner"
                )

    @property
    def extent(self) -> np.ndarray:
        return np.asarray(self.max, dtype=np.float64) - np.asarray(self.min, dtype=np.float64)

    @property
    def volume(self) -> float:
        return float(np.prod(self.extent))

    @property
    def center(self) -> np.ndarray:
        return (np.asarray(self.min, dtype=np.float64) + np.asarray(self.max)) / 2.0

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask over an (N, 3) array of points. Bounds are inclusive."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise BoxError(f"expected an (N, 3) array of points, got shape {points.shape}")
        lo = np.asarray(self.min, dtype=np.float64)
        hi = np.asarray(self.max, dtype=np.float64)
        return np.all((points >= lo) & (points <= hi), axis=1)

    def expanded(self, margin: float) -> BoundingBox:
        """A box grown outward by `margin` on every side."""
        delta = np.full(3, float(margin))
        lo = np.asarray(self.min, dtype=np.float64) - delta
        hi = np.asarray(self.max, dtype=np.float64) + delta
        if np.any(hi < lo):
            raise BoxError(
                f"margin {margin} is more negative than half of box {self.name!r}'s smallest "
                f"side ({self.extent.min()}); the box would invert"
            )
        return BoundingBox(self.name, tuple(lo.tolist()), tuple(hi.tolist()))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "min": list(self.min), "max": list(self.max)}

    @classmethod
    def from_dict(cls, raw: object, index: int = 0) -> BoundingBox:
        if not isinstance(raw, dict):
            raise BoxError(f"box #{index} is not a mapping")
        missing = [key for key in ("name", "min", "max") if key not in raw]
        if missing:
            raise BoxError(f"box #{index} is missing {', '.join(missing)}")
        name = str(raw["name"])
        corners = []
        for key in ("min", "max"):
            value = raw[key]
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise BoxError(f"box {name!r}: {key} must be a list of three numbers")
            try:
                corners.append(tuple(float(v) for v in value))
            except (TypeError, ValueError) as exc:
                raise BoxError(f"box {name!r}: {key} is not numeric ({exc})") from None
        return cls(name=name, min=corners[0], max=corners[1])


def load_boxes(path: Path) -> list[BoundingBox]:
    """Load named boxes from YAML.

    Expected shape::

        boxes:
          - name: kitchen
            min: [-2.0, -1.0, 0.0]
            max: [ 3.0,  2.5, 2.8]
    """
    path = Path(path)
    if not path.is_file():
        raise BoxError(f"Box file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict) or "boxes" not in raw:
        raise BoxError(f"{path}: expected a top-level 'boxes:' list")
    entries = raw["boxes"]
    if not isinstance(entries, list) or not entries:
        raise BoxError(f"{path}: 'boxes' must be a non-empty list")

    boxes = [BoundingBox.from_dict(entry, i) for i, entry in enumerate(entries)]

    seen: dict[str, int] = {}
    for i, box in enumerate(boxes):
        if box.name in seen:
            raise BoxError(
                f"{path}: duplicate box name {box.name!r} (entries #{seen[box.name]} and #{i}); "
                f"names become output directories and must be unique"
            )
        seen[box.name] = i
    return boxes


def save_boxes(path: Path, boxes: list[BoundingBox], header: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        {"boxes": [box.to_dict() for box in boxes]}, sort_keys=False, default_flow_style=None
    )
    path.write_text((f"# {header}\n" if header else "") + body)


def bounds_of(points: np.ndarray, percentile: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of a point set, optionally trimming outliers.

    A single stray triangulated point can sit far outside a room and inflate
    its box enormously, so `--suggest` trims by percentile rather than using
    the raw extremes.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        raise BoxError("cannot compute bounds of an empty point set")
    if percentile <= 0:
        return points.min(axis=0), points.max(axis=0)
    lo = np.percentile(points, percentile, axis=0)
    hi = np.percentile(points, 100.0 - percentile, axis=0)
    return lo, hi
