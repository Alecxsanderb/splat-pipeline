"""Reader and writer for 3D Gaussian Splatting PLY files.

Deliberately generic about properties. A 3DGS splat carries position,
normals, DC and higher-order spherical-harmonic coefficients, opacity, scale
and a rotation quaternion -- and the number of SH coefficients depends on the
degree the model was trained at (45 `f_rest_*` fields at degree 3, fewer at
lower degrees). Rather than hardcode a schema and silently drop whatever does
not match, this module reads whatever properties the header declares, carries
them through as a numpy structured array, and writes them back in the same
order with the same types. Cropping and concatenation are then just row
operations, so no property can be lost by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PlyError(RuntimeError):
    """Raised when a PLY file cannot be read or is not a usable splat."""


# PLY scalar type names -> numpy base types. Both the canonical and the
# short/alias spellings appear in the wild.
_PLY_TO_NUMPY = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}

_NUMPY_TO_PLY = {
    "i1": "char", "u1": "uchar",
    "i2": "short", "u2": "ushort",
    "i4": "int", "u4": "uint",
    "f4": "float", "f8": "double",
}

POSITION_FIELDS = ("x", "y", "z")


@dataclass
class PlyHeader:
    fmt: str  # "binary_little_endian" | "binary_big_endian" | "ascii"
    element: str
    count: int
    properties: list[tuple[str, str]]  # (name, ply type name), in file order
    comments: list[str]
    header_bytes: int


@dataclass(eq=False)
class GaussianCloud:
    """A 3DGS point cloud plus the exact property schema it was read with."""

    data: np.ndarray  # structured array, one record per Gaussian
    properties: list[tuple[str, str]]  # (name, ply type), file order preserved
    comments: list[str]

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def property_names(self) -> list[str]:
        return [name for name, _ in self.properties]

    @property
    def xyz(self) -> np.ndarray:
        """(N, 3) float64 positions."""
        return np.stack(
            [self.data[axis].astype(np.float64) for axis in POSITION_FIELDS], axis=1
        )

    def select(self, mask: np.ndarray) -> GaussianCloud:
        """A new cloud holding only the rows where `mask` is true."""
        return GaussianCloud(
            data=self.data[mask], properties=list(self.properties), comments=list(self.comments)
        )


def _parse_header(handle) -> PlyHeader:
    magic = handle.readline()
    if magic.strip() != b"ply":
        raise PlyError("not a PLY file (missing 'ply' magic)")

    fmt: str | None = None
    comments: list[str] = []
    elements: list[tuple[str, int, list[tuple[str, str]]]] = []

    while True:
        raw = handle.readline()
        if not raw:
            raise PlyError("PLY header ended without 'end_header'")
        line = raw.decode("ascii", errors="replace").strip()
        if not line:
            continue
        parts = line.split()
        keyword = parts[0]

        if keyword == "format":
            fmt = parts[1]
            if fmt not in ("binary_little_endian", "binary_big_endian", "ascii"):
                raise PlyError(f"unsupported PLY format {fmt!r}")
        elif keyword == "comment":
            comments.append(line[len("comment"):].strip())
        elif keyword == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif keyword == "property":
            if not elements:
                raise PlyError("PLY property declared before any element")
            if parts[1] == "list":
                # 3DGS splats have no list properties; supporting them would
                # mean variable-stride records, and guessing here would corrupt
                # every downstream row operation.
                raise PlyError(
                    f"list property {parts[-1]!r} is not supported; this does not look "
                    f"like a 3D Gaussian Splatting point cloud"
                )
            if parts[1] not in _PLY_TO_NUMPY:
                raise PlyError(f"unknown PLY property type {parts[1]!r}")
            elements[-1][2].append((parts[2], parts[1]))
        elif keyword == "end_header":
            break

    if fmt is None:
        raise PlyError("PLY header has no 'format' line")
    if not elements:
        raise PlyError("PLY header declares no elements")

    vertex = next((e for e in elements if e[0] == "vertex"), elements[0])
    if len(elements) > 1:
        others = [e[0] for e in elements if e[0] != vertex[0]]
        logger.warning(
            "PLY declares element(s) %s besides %r; only %r is read",
            ", ".join(others), vertex[0], vertex[0],
        )
    if not vertex[2]:
        raise PlyError(f"PLY element {vertex[0]!r} declares no properties")

    return PlyHeader(
        fmt=fmt,
        element=vertex[0],
        count=vertex[1],
        properties=vertex[2],
        comments=comments,
        header_bytes=handle.tell(),
    )


def _structured_dtype(properties: list[tuple[str, str]], fmt: str) -> np.dtype:
    order = ">" if fmt == "binary_big_endian" else "<"
    return np.dtype([(name, order + _PLY_TO_NUMPY[ply_type]) for name, ply_type in properties])


def read_ply(path: Path) -> GaussianCloud:
    """Read a splat PLY, preserving every property exactly as declared."""
    path = Path(path)
    if not path.is_file():
        raise PlyError(f"PLY file not found: {path}")

    with path.open("rb") as handle:
        header = _parse_header(handle)
        dtype = _structured_dtype(header.properties, header.fmt)

        if header.fmt == "ascii":
            rows = []
            for _ in range(header.count):
                line = handle.readline()
                if not line:
                    raise PlyError(f"{path}: PLY ended after {len(rows)} of {header.count} rows")
                rows.append(tuple(float(v) for v in line.split()))
            data = np.array(rows, dtype=dtype) if rows else np.zeros(0, dtype=dtype)
        else:
            payload = handle.read(dtype.itemsize * header.count)
            if len(payload) < dtype.itemsize * header.count:
                raise PlyError(
                    f"{path}: truncated PLY body -- header declares {header.count} vertices "
                    f"({dtype.itemsize * header.count} bytes) but only {len(payload)} remain"
                )
            data = np.frombuffer(payload, dtype=dtype, count=header.count)

    missing = [axis for axis in POSITION_FIELDS if axis not in data.dtype.names]
    if missing:
        raise PlyError(
            f"{path}: PLY has no {'/'.join(missing)} propert(y/ies); cannot treat it as a "
            f"point cloud"
        )

    return GaussianCloud(
        data=np.ascontiguousarray(data),
        properties=list(header.properties),
        comments=list(header.comments),
    )


def write_ply(path: Path, cloud: GaussianCloud, *, comments: list[str] | None = None) -> None:
    """Write a splat PLY as binary_little_endian, preserving property order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dtype = _structured_dtype(cloud.properties, "binary_little_endian")
    payload = cloud.data.astype(dtype, copy=False) if cloud.data.dtype != dtype else cloud.data

    lines = ["ply", "format binary_little_endian 1.0"]
    for comment in list(cloud.comments) + list(comments or []):
        lines.append(f"comment {comment}")
    lines.append(f"element vertex {len(cloud)}")
    for name, ply_type in cloud.properties:
        lines.append(f"property {ply_type} {name}")
    lines.append("end_header")

    with path.open("wb") as handle:
        handle.write(("\n".join(lines) + "\n").encode("ascii"))
        handle.write(payload.tobytes())


def concatenate(clouds: list[GaussianCloud]) -> GaussianCloud:
    """Concatenate clouds that share an identical property schema.

    Schemas must match exactly: splats trained at different SH degrees carry
    different numbers of `f_rest_*` fields, and silently padding or truncating
    them would corrupt the appearance of whichever chunk was coerced.
    """
    if not clouds:
        raise PlyError("nothing to concatenate")

    reference = clouds[0].properties
    for i, cloud in enumerate(clouds[1:], start=1):
        if cloud.properties != reference:
            expected = {n for n, _ in reference}
            got = {n for n, _ in cloud.properties}
            detail = ""
            if expected != got:
                only_first = sorted(expected - got)[:5]
                only_other = sorted(got - expected)[:5]
                detail = (
                    f" (first has {len(expected)} properties, this one {len(got)};"
                    f" only in first: {only_first or 'none'};"
                    f" only in this: {only_other or 'none'})"
                )
            raise PlyError(
                f"cloud {i} has a different property schema than the first{detail}. "
                f"These splats were probably trained with different spherical-harmonic "
                f"degrees and cannot be merged without losing appearance data."
            )

    dtype = _structured_dtype(reference, "binary_little_endian")
    merged = np.concatenate([c.data.astype(dtype, copy=False) for c in clouds])
    return GaussianCloud(
        data=merged,
        properties=list(reference),
        comments=list(clouds[0].comments),
    )
