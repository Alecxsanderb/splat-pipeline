"""Top-down camera-position plot for eyeballing a reconstruction.

COLMAP's world frame is arbitrary, so "top-down" is not simply the XY or XZ
plane. Camera centres are projected onto their two dominant principal axes:
for a walkthrough capture the cameras sit at roughly constant height, so those
two axes span the floor plane and the result reads as a floor plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlotResult:
    status: str  # "written" | "skipped"
    path: Path | None = None
    reason: str | None = None
    planarity: float | None = None
    height_spread: float | None = None
    basis: list[list[float]] = field(default_factory=list)


def floor_plane_projection(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera centres onto their two dominant principal axes.

    Returns (xy, height, basis). The sign of each principal axis is pinned so
    repeated runs produce the same orientation instead of randomly mirroring.
    """
    centered = centers - centers.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)

    for i in range(vt.shape[0]):
        if vt[i][np.argmax(np.abs(vt[i]))] < 0:
            vt[i] = -vt[i]

    return centered @ vt[:2].T, centered @ vt[2], vt


def render_top_down(
    centers: np.ndarray,
    groups: list[str],
    labels: np.ndarray | None,
    output_path: Path,
    *,
    clip_series: dict[str, list[int]] | None = None,
    dpi: int = 150,
    title_suffix: str = "",
) -> PlotResult:
    """Write a top-down scatter of camera positions, coloured by source group.

    Never raises: a missing plot is not a failed reconstruction, so every
    failure path degrades to a skip with a recorded reason.
    """
    if centers.shape[0] < 3:
        reason = f"only {centers.shape[0]} camera(s); need at least 3"
        logger.warning("Skipping top-down plot: %s", reason)
        return PlotResult(status="skipped", reason=reason)

    xy, height, basis = floor_plane_projection(centers)
    centered = centers - centers.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)

    if singular[1] < 1e-9 * max(singular[0], 1e-30):
        reason = "camera centres are collinear; a top-down view is not meaningful"
        logger.warning("Skipping top-down plot: %s", reason)
        return PlotResult(status="skipped", reason=reason)

    planarity = float(singular[2] / singular[1]) if singular[1] > 0 else 0.0
    height_spread = float(np.percentile(height, 95) - np.percentile(height, 5))
    if planarity > 0.5:
        logger.warning(
            "Camera positions are not planar (ratio %.2f); the top-down view may be "
            "misleading -- this is expected for a multi-storey capture, but can also "
            "mean the reconstruction is warped", planarity,
        )

    try:
        import matplotlib

        matplotlib.use("Agg")  # must precede the pyplot import on a headless host
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logger.warning(
            "matplotlib not available (%s); skipping the top-down plot "
            "(pip install matplotlib)", exc,
        )
        return PlotResult(status="skipped", reason="matplotlib_missing")

    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        unique_groups = sorted(set(groups))
        colormap = plt.get_cmap("tab20" if len(unique_groups) > 10 else "tab10")
        color_of = {g: colormap(i % colormap.N) for i, g in enumerate(unique_groups)}
        markers = ["o", "x", "^", "s", "D", "v"]

        # A faint line through each clip in capture order is what turns a point
        # cloud into something recognisable as a walking path through rooms.
        for indices in (clip_series or {}).values():
            if len(indices) > 1:
                ax.plot(xy[indices, 0], xy[indices, 1], lw=0.4, alpha=0.3, color="grey", zorder=1)

        for group in unique_groups:
            member = np.array([g == group for g in groups])
            if labels is None:
                ax.scatter(
                    xy[member, 0], xy[member, 1], s=8, alpha=0.75,
                    color=color_of[group], label=group, zorder=2,
                )
                continue
            # Marker encodes component, so fragmentation is visible at a glance.
            for component in sorted(set(labels[member].tolist())):
                sel = member & (labels == component)
                if not sel.any():
                    continue
                suffix = "" if component == 0 else f" (component {component})"
                ax.scatter(
                    xy[sel, 0], xy[sel, 1], s=8, alpha=0.75, color=color_of[group],
                    marker=markers[min(component, len(markers) - 1)],
                    label=f"{group}{suffix}", zorder=2,
                )

        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_xlabel("principal axis 1 (model units, scale arbitrary)")
        ax.set_ylabel("principal axis 2 (model units, scale arbitrary)")
        ax.set_title(
            f"Camera positions, top-down{title_suffix}\n"
            f"{centers.shape[0]} cameras, planarity {planarity:.2f}"
        )
        ax.legend(loc="best", fontsize="small", markerscale=2)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - a plot must never fail verification
        logger.warning("Could not render the top-down plot: %s", exc)
        return PlotResult(status="skipped", reason=str(exc))

    return PlotResult(
        status="written",
        path=output_path,
        planarity=planarity,
        height_spread=height_spread,
        basis=basis.tolist(),
    )
