"""verify: sanity-check a COLMAP reconstruction and say plainly whether it worked.

Exit codes:
    0  every fail-severity check passed (warnings may still be present)
    1  verification ran and something is wrong with the reconstruction
    3  verification could not run at all (nothing to verify yet)

Exit 3 is kept distinct from 1 because "your reconstruction is bad" and "you
haven't run sfm yet" call for completely different responses.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pipeline.colmap_model import (
    ColmapModelError,
    Model,
    find_models,
    read_model,
)
from pipeline.commands.select import FRAME_NAME_RE
from pipeline.config import PipelineConfig
from pipeline.connectivity import (
    BoundaryPair,
    component_threshold_curve,
    connected_components,
    count_shared_points,
    cross_component_pairs,
    spatial_candidates,
)
from pipeline.plotting import PlotResult, render_top_down

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_UNVERIFIABLE = 3

UNGROUPED = "<ungrouped>"


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("verify", help="Verify a COLMAP reconstruction")
    # Every argument must stay optional: `splat verify` with no flags is the
    # normal invocation and is exercised by the CLI tests.
    parser.add_argument("--sparse-dir", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--model", type=int, default=None,
                        help="Analyze this sparse submodel instead of the largest")
    parser.add_argument("--min-registration", type=float, default=None)
    parser.add_argument("--min-common-points", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser


@dataclass(frozen=True)
class VerifyOptions:
    sparse_dir: Path
    images_dir: Path
    database_path: Path
    report_dir: Path
    model_index: int | None
    min_registration_rate: float
    warn_registration_rate: float
    min_common_points: int
    min_component_size: int
    min_observations_per_image: int
    warn_mean_reprojection_error: float
    component_thresholds: tuple[int, ...]
    max_track_length: int
    max_pair_candidates: int
    pairs_per_component_pair: int
    spatial_neighbors_per_image: int
    max_view_angle_deg: float
    image_extensions: tuple[str, ...]
    plot: bool
    plot_dpi: int


def _resolve_options(args: argparse.Namespace, config: PipelineConfig) -> VerifyOptions:
    cfg = config.verify

    def pick(name: str, fallback):
        value = getattr(args, name, None)
        return fallback if value is None else value

    return VerifyOptions(
        sparse_dir=pick("sparse_dir", config.paths.output_dir / "sparse"),
        images_dir=pick("images_dir", config.paths.output_dir / "images"),
        database_path=config.paths.workdir / "sfm" / "database.db",
        report_dir=config.paths.workdir / "verify",
        model_index=getattr(args, "model", None),
        min_registration_rate=pick("min_registration", cfg.min_registration_rate),
        warn_registration_rate=cfg.warn_registration_rate,
        min_common_points=pick("min_common_points", cfg.min_common_points),
        min_component_size=cfg.min_component_size,
        min_observations_per_image=cfg.min_observations_per_image,
        warn_mean_reprojection_error=cfg.warn_mean_reprojection_error,
        component_thresholds=tuple(cfg.component_thresholds),
        max_track_length=cfg.max_track_length,
        max_pair_candidates=cfg.max_pair_candidates,
        pairs_per_component_pair=cfg.pairs_per_component_pair,
        spatial_neighbors_per_image=cfg.spatial_neighbors_per_image,
        max_view_angle_deg=cfg.max_view_angle_deg,
        image_extensions=tuple(cfg.image_extensions),
        plot=cfg.plot and not getattr(args, "no_plot", False),
        plot_dpi=cfg.plot_dpi,
    )


@dataclass
class Check:
    name: str
    severity: str  # "fail" | "warn"
    passed: bool
    detail: str
    remediation: str = ""


@dataclass
class Inventory:
    total: int = 0
    per_group: dict[str, int] = field(default_factory=dict)
    stray_files: list[str] = field(default_factory=list)
    database_images: int | None = None


def _clip_of(name: str) -> str:
    """Clip (capture pass) a frame came from, or the group for a still photo.

    Qualified by group, because two rooms recorded to identically named files
    (`room1/clip.mp4`, `room2/clip.mp4`) both yield `clip_000123.jpg` and would
    otherwise be reported -- and plotted -- as a single capture pass.
    """
    group = name.split("/")[0] if "/" in name else UNGROUPED
    match = FRAME_NAME_RE.match(Path(name).stem)
    return f"{group}/{match.group('clip')}" if match else group


def inventory_input_images(images_root: Path, extensions: tuple[str, ...]) -> Inventory:
    """Count the images COLMAP was actually pointed at.

    This is the denominator for the registration rate: images.bin only ever
    contains registered images, so it can never tell us how many were offered.
    """
    inventory = Inventory()
    if not images_root.is_dir():
        return inventory

    suffixes = {ext.lower() for ext in extensions}
    per_group: dict[str, int] = defaultdict(int)
    for path in sorted(images_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        relative = path.relative_to(images_root)
        # organize writes images/<group>/<file>; anything else still counts
        # (COLMAP recurses, so it was offered too) but is surfaced as an anomaly
        # rather than silently inflating a group's total.
        if len(relative.parts) == 2:
            per_group[relative.parts[0]] += 1
        else:
            per_group[UNGROUPED] += 1
            inventory.stray_files.append(relative.as_posix())
        inventory.total += 1

    inventory.per_group = dict(per_group)
    return inventory


def count_database_images(database_path: Path) -> int | None:
    """Images COLMAP's feature extractor actually ingested, or None.

    Splitting 'on disk' from 'in database' from 'registered' separates two very
    different failures: images COLMAP never saw versus images it could not
    place. Best-effort only -- never affects the verdict.
    """
    if not database_path.is_file():
        return None
    try:
        uri = f"file:{database_path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "images" not in tables:
                return None
            return int(connection.execute("SELECT COUNT(*) FROM images").fetchone()[0])
    except (sqlite3.Error, OSError) as exc:
        logger.debug("Could not read %s: %s", database_path, exc)
        return None


def matched_pairs_from_database(
    database_path: Path, names: set[str]
) -> set[tuple[str, str]] | None:
    """Image-name pairs that already have a verified two-view geometry.

    A pair that already shares 3D points has already been matched, so feeding
    it to matches_importer would accomplish nothing; those get reported as
    diagnostic-only instead of as a bridge candidate.
    """
    if not database_path.is_file():
        return None
    try:
        uri = f"file:{database_path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "images" not in tables:
                return None
            table = "two_view_geometries" if "two_view_geometries" in tables else "matches"
            if table not in tables:
                return None
            # Database ids are not model ids; go via the name.
            name_of = {
                int(row[0]): row[1]
                for row in connection.execute("SELECT image_id, name FROM images")
            }
            matched: set[tuple[str, str]] = set()
            for (pair_id,) in connection.execute(f"SELECT pair_id FROM {table}"):
                image_id2 = int(pair_id) % 2147483647
                image_id1 = (int(pair_id) - image_id2) // 2147483647
                a, b = name_of.get(image_id1), name_of.get(image_id2)
                if a in names and b in names:
                    matched.add((min(a, b), max(a, b)))
            return matched
    except (sqlite3.Error, OSError) as exc:
        logger.debug("Could not read pairs from %s: %s", database_path, exc)
        return None


@dataclass
class ErrorStats:
    available: bool
    reason: str = ""
    mean: float = 0.0
    median: float = 0.0
    p90: float = 0.0
    maximum: float = 0.0
    observation_weighted_mean: float = 0.0


def reprojection_error_stats(model: Model) -> ErrorStats:
    """Statistics over the per-point error stored in points3D.bin.

    That field already is each point's mean reprojection error and is the
    number COLMAP itself reports. Recomputing would mean reimplementing all
    eleven distortion models, where a sign or parameter-order slip produces a
    plausible-looking wrong number in the one command whose job is telling the
    user whether to trust the reconstruction.
    """
    errors = model.points3d.error
    if errors.size == 0:
        return ErrorStats(available=False, reason="model contains no 3D points")

    finite = errors[np.isfinite(errors)]
    if finite.size == 0 or np.all(finite <= 0):
        # GLOMAP can leave this field at 0 or -1 when no bundle-adjustment pass
        # filled it. Reporting "mean error 0.00 px" would read as perfect.
        return ErrorStats(
            available=False,
            reason="points3D.bin carries no usable reprojection error "
                   "(all values are zero, negative or non-finite)",
        )

    positive = finite[finite > 0]
    lengths = model.points3d.track_lengths().astype(np.float64)
    weighted = float(np.sum(errors * lengths) / np.sum(lengths)) if np.sum(lengths) else 0.0
    return ErrorStats(
        available=True,
        mean=float(np.mean(positive)),
        median=float(np.median(positive)),
        p90=float(np.percentile(positive, 90)),
        maximum=float(np.max(positive)),
        observation_weighted_mean=weighted,
    )


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0, "p05": 0, "median": 0, "mean": 0, "p95": 0, "max": 0}
    return {
        "min": int(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": int(np.max(values)),
    }


def _registration_by_key(
    model: Model, totals: dict[str, int], key_of
) -> list[dict[str, object]]:
    registered: dict[str, int] = defaultdict(int)
    for image in model.images.values():
        registered[key_of(image.name)] += 1

    rows = []
    for key in sorted(set(totals) | set(registered)):
        total = totals.get(key, 0)
        got = registered.get(key, 0)
        rows.append({
            "key": key,
            "total": total,
            "registered": got,
            "rate": (got / total) if total else 0.0,
        })
    rows.sort(key=lambda r: (r["rate"], -r["total"]))
    return rows


def _unverifiable(options: VerifyOptions, reason: str) -> int:
    logger.error("Cannot verify: %s", reason)
    options.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = options.report_dir / "verify_report.json"
    report_path.write_text(json.dumps({
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "unverifiable",
        "exit_code": EXIT_UNVERIFIABLE,
        "reason": reason,
    }, indent=2) + "\n")
    logger.error("Wrote %s", report_path)
    return EXIT_UNVERIFIABLE


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _select_candidates(
    boundary: list[BoundaryPair], options: VerifyOptions
) -> list[BoundaryPair]:
    """Cap candidates round-robin so one dominant component pair can't starve others."""
    grouped: dict[tuple[int, int], list[BoundaryPair]] = defaultdict(list)
    for pair in boundary:
        grouped[(pair.component_a, pair.component_b)].append(pair)
    for pairs in grouped.values():
        del pairs[options.pairs_per_component_pair:]

    selected: list[BoundaryPair] = []
    queues = list(grouped.values())
    index = 0
    while queues and len(selected) < options.max_pair_candidates:
        queue = queues[index % len(queues)]
        if queue:
            selected.append(queue.pop(0))
            index += 1
        else:
            queues.remove(queue)
    return selected


def run(args: argparse.Namespace, config: PipelineConfig) -> int:
    options = _resolve_options(args, config)
    options.report_dir.mkdir(parents=True, exist_ok=True)

    inventory = inventory_input_images(options.images_dir, options.image_extensions)
    if inventory.total == 0:
        return _unverifiable(
            options, f"no input images found under {options.images_dir}; run `splat organize` first"
        )
    inventory.database_images = count_database_images(options.database_path)

    try:
        model_paths = find_models(options.sparse_dir)
    except ColmapModelError as exc:
        return _unverifiable(options, str(exc))

    try:
        models = [read_model(path, read_points2d=False) for path in model_paths]
    except ColmapModelError as exc:
        return _unverifiable(options, f"could not read the sparse model: {exc}")

    if options.model_index is not None:
        chosen = [
            (p, m) for p, m in zip(model_paths, models, strict=True)
            if p.name == str(options.model_index)
        ]
        if not chosen:
            return _unverifiable(
                options, f"no submodel named {options.model_index} in {options.sparse_dir}"
            )
        primary_path, primary = chosen[0]
    else:
        primary_path, primary = max(
            zip(model_paths, models, strict=True), key=lambda pair: pair[1].num_registered_images
        )

    checks: list[Check] = []
    report: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "sparse_dir": options.sparse_dir.as_posix(),
        "images_dir": options.images_dir.as_posix(),
        "inventory": {
            "total_images_on_disk": inventory.total,
            "images_in_database": inventory.database_images,
            "per_group": inventory.per_group,
            "stray_files": inventory.stray_files[:50],
        },
        "models": [
            {
                "path": path.as_posix(),
                "registered_images": model.num_registered_images,
                "points": model.num_points3d,
                "observations": model.num_observations,
                "is_primary": path == primary_path,
            }
            for path, model in zip(model_paths, models, strict=True)
        ],
    }

    logger.info("Input images on disk: %d", inventory.total)
    if inventory.database_images is not None:
        logger.info(
            "Images in COLMAP database: %d (%d never reached the database)",
            inventory.database_images, inventory.total - inventory.database_images,
        )
        if inventory.database_images < inventory.total:
            checks.append(Check(
                "database_gap", "warn", False,
                f"{inventory.total - inventory.database_images} image(s) on disk never "
                f"reached the COLMAP database",
                "These failed during feature_extractor (unreadable file or duplicate name); "
                "check the sfm log.",
            ))
    if inventory.stray_files:
        checks.append(Check(
            "stray_images", "warn", False,
            f"{len(inventory.stray_files)} file(s) are not laid out as images/<group>/<file>",
            "Re-run `splat organize` so COLMAP sees one camera model per folder.",
        ))

    # --- models -----------------------------------------------------------
    logger.info("Found %d model(s) in %s", len(models), options.sparse_dir)
    for path, model in zip(model_paths, models, strict=True):
        marker = " (primary)" if path == primary_path else ""
        logger.info(
            "  %-24s %6d images  %8d points%s",
            path.name, model.num_registered_images, model.num_points3d, marker,
        )
    checks.append(Check(
        "single_model", "fail", len(models) == 1,
        f"{len(models)} separate model(s) in {options.sparse_dir}"
        + ("" if len(models) == 1 else " — the scene fragmented into separate reconstructions"),
        "" if len(models) == 1 else
        "Bridge the models by re-matching the smaller one against the rest, then re-run the "
        "mapper. Per-model image lists were written next to this report.",
    ))

    for path, model in zip(model_paths, models, strict=True):
        if path == primary_path:
            continue
        listing = options.report_dir / f"model_{path.name}_images.txt"
        listing.write_text(
            "\n".join(sorted(img.name for img in model.images.values())) + "\n"
        )

    # --- registration -----------------------------------------------------
    registered = primary.num_registered_images
    rate = registered / inventory.total
    union = len({img.name for model in models for img in model.images.values()})
    logger.info(
        "Registered in primary model: %d / %d (%.1f%%)", registered, inventory.total, rate * 100
    )
    if len(models) > 1:
        logger.info(
            "Registered across all models: %d / %d (%.1f%%) — informational only, not used "
            "for pass/fail", union, inventory.total, union / inventory.total * 100,
        )

    checks.append(Check(
        "non_empty", "fail", registered > 0 and primary.num_points3d > 0,
        f"primary model has {registered} image(s) and {primary.num_points3d} point(s)",
        "The mapper produced an empty reconstruction; check the sfm log for matching failures.",
    ))
    checks.append(Check(
        "registration_rate", "fail", rate >= options.min_registration_rate,
        f"{rate * 100:.1f}% of input images registered "
        f"(threshold {options.min_registration_rate * 100:.0f}%)",
        "Add coverage where images dropped out, or lower select.blur_threshold so sharper "
        "overlapping frames survive selection.",
    ))
    if rate >= options.min_registration_rate:
        comfortable = rate >= options.warn_registration_rate
        checks.append(Check(
            "registration_rate_warn", "warn", comfortable,
            f"{rate * 100:.1f}% registered "
            f"({'at or above' if comfortable else 'below'} the "
            f"{options.warn_registration_rate * 100:.0f}% comfort level)",
            "Usable, but inspect registration_by_group.csv for where the losses are.",
        ))

    group_rows = _registration_by_key(
        primary, inventory.per_group, lambda name: name.split("/")[0] if "/" in name else UNGROUPED
    )
    # Clip totals come from the on-disk listing, so clips that registered
    # nothing at all still show up in the table.
    clip_totals: dict[str, int] = defaultdict(int)
    if options.images_dir.is_dir():
        suffixes = {e.lower() for e in options.image_extensions}
        for image_path in options.images_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in suffixes:
                rel = image_path.relative_to(options.images_dir).as_posix()
                clip_totals[_clip_of(rel)] += 1
    clip_rows = _registration_by_key(primary, dict(clip_totals), _clip_of)

    logger.info("Registration by source folder (worst first):")
    for row in group_rows:
        logger.info(
            "  %-24s %5d / %-5d  %5.1f%%",
            row["key"], row["registered"], row["total"], row["rate"] * 100,
        )
    dead_groups = [r["key"] for r in group_rows if r["total"] > 0 and r["registered"] == 0]
    checks.append(Check(
        "group_coverage", "fail", not dead_groups,
        "every source folder registered at least one image" if not dead_groups
        else f"source folder(s) with zero registered images: {', '.join(dead_groups)}",
        "" if not dead_groups else
        "That room or camera contributed nothing. Check its frames are sharp and overlap "
        "the rest of the capture.",
    ))
    weak_clips = [r["key"] for r in clip_rows if r["total"] >= 10 and r["rate"] < 0.5]
    if weak_clips:
        checks.append(Check(
            "clip_coverage", "warn", False,
            f"{len(weak_clips)} clip(s) registered under 50%: {', '.join(weak_clips[:5])}",
            "Those passes may be too blurry or too weakly overlapped with the rest.",
        ))

    # --- error, tracks, observations ---------------------------------------
    errors = reprojection_error_stats(primary)
    if errors.available:
        logger.info(
            "Reprojection error: mean %.3f px, median %.3f px, p90 %.3f px, max %.3f px",
            errors.mean, errors.median, errors.p90, errors.maximum,
        )
        logger.info(
            "  observation-weighted mean %.3f px (per-point values from points3D.bin)",
            errors.observation_weighted_mean,
        )
        checks.append(Check(
            "reprojection_error", "warn",
            errors.mean <= options.warn_mean_reprojection_error,
            f"mean reprojection error {errors.mean:.3f} px "
            f"(warn above {options.warn_mean_reprojection_error} px)",
            "High residuals suggest bad intrinsics or mismatches; consider re-running sfm "
            "with a fixed camera model per folder.",
        ))
    else:
        logger.warning("Reprojection error unavailable: %s", errors.reason)
        checks.append(Check(
            "reprojection_error", "warn", False,
            f"reprojection error unavailable — {errors.reason}",
            "The mapper did not populate this field; treat pose quality as unverified.",
        ))

    lengths = primary.points3d.track_lengths()
    obs_per_image = np.array(list(primary.observations_per_image().values()))
    short_tracks = float(np.mean(lengths <= 2)) if lengths.size else 0.0
    logger.info(
        "Track length: mean %.2f, median %.1f (%.0f%% of points seen by only 2 images)",
        primary.mean_track_length(), float(np.median(lengths)) if lengths.size else 0.0,
        short_tracks * 100,
    )
    distribution = _percentiles(obs_per_image)
    logger.info(
        "Observations per image: min %d, p05 %.0f, median %.0f, mean %.1f, p95 %.0f, max %d",
        distribution["min"], distribution["p05"], distribution["median"],
        distribution["mean"], distribution["p95"], distribution["max"],
    )
    weak = int(np.count_nonzero(obs_per_image < options.min_observations_per_image))
    if weak:
        checks.append(Check(
            "weak_images", "warn", False,
            f"{weak} registered image(s) contribute fewer than "
            f"{options.min_observations_per_image} observations",
            "These are held in place by very little evidence; their poses are the least "
            "trustworthy in the model.",
        ))

    # --- extent ------------------------------------------------------------
    image_ids, centers = primary.camera_centers()
    extent = centers.max(axis=0) - centers.min(axis=0) if centers.size else np.zeros(3)
    logger.info(
        "Camera bounding box (model units, scale arbitrary): "
        "x %.2f, y %.2f, z %.2f; diagonal %.2f",
        extent[0], extent[1], extent[2], float(np.linalg.norm(extent)),
    )

    # --- connectivity ------------------------------------------------------
    dense_ids = np.array(image_ids, dtype=np.int64)
    pairs = count_shared_points(
        dense_ids,
        primary.points3d.track_image_ids,
        primary.points3d.track_offsets,
        max_track_length=options.max_track_length,
    )
    labels = connected_components(pairs, options.min_common_points)
    unique_labels, sizes = (
        np.unique(labels, return_counts=True) if labels.size else (np.array([]), np.array([]))
    )
    big = [int(s) for s in sizes if s >= options.min_component_size]
    logger.info(
        "Connectivity at >=%d shared points: %d component(s) %s",
        options.min_common_points, len(sizes), sorted((int(s) for s in sizes), reverse=True)[:10],
    )
    curve = component_threshold_curve(pairs, options.component_thresholds)
    logger.info(
        "  components by threshold: %s",
        ", ".join(f"N={t}: {c}" for t, c in curve),
    )
    small = len(sizes) - len(big)
    if small:
        logger.warning(
            "  %d component(s) smaller than %d image(s) are reported but do not fail the run",
            small, options.min_component_size,
        )
    checks.append(Check(
        "connectivity", "fail", len(big) <= 1,
        f"{len(big)} connected component(s) of at least {options.min_component_size} images "
        f"at >={options.min_common_points} shared points",
        "" if len(big) <= 1 else
        "The scene is in disconnected pieces. Candidate bridging pairs were written to "
        "candidate_pairs.txt.",
    ))

    # --- boundary candidates -----------------------------------------------
    boundary: list[BoundaryPair] = []
    if len(unique_labels) > 1:
        boundary = cross_component_pairs(pairs, labels, options.min_common_points)
        directions = primary.viewing_directions(image_ids)
        connected_by_points = {(p.component_a, p.component_b) for p in boundary}
        for i, comp_a in enumerate(unique_labels.tolist()):
            for comp_b in unique_labels.tolist()[i + 1:]:
                if (comp_a, comp_b) in connected_by_points:
                    continue
                boundary.extend(spatial_candidates(
                    centers, directions, labels, comp_a, comp_b,
                    neighbors_per_image=options.spatial_neighbors_per_image,
                    max_view_angle_deg=options.max_view_angle_deg,
                ))

    name_of = {i: primary.images[image_id].name for i, image_id in enumerate(image_ids)}
    already = matched_pairs_from_database(options.database_path, set(name_of.values()))

    candidate_rows = []
    unmatched: list[BoundaryPair] = []
    for pair in boundary:
        name_a, name_b = name_of[pair.dense_a], name_of[pair.dense_b]
        key = (min(name_a, name_b), max(name_a, name_b))
        is_matched = None if already is None else (key in already)
        candidate_rows.append([
            name_a, name_b, pair.component_a, pair.component_b, pair.shared_points,
            "" if pair.distance is None else f"{pair.distance:.4f}",
            "" if pair.view_angle_deg is None else f"{pair.view_angle_deg:.1f}",
            "" if is_matched is None else is_matched, pair.source,
        ])
        if not is_matched:
            unmatched.append(pair)

    selected = _select_candidates(unmatched, options)
    pair_file = options.report_dir / "candidate_pairs.txt"
    pair_lines = sorted(
        f"{name_of[p.dense_a]} {name_of[p.dense_b]}" for p in selected
    )
    pair_file.write_text("\n".join(pair_lines) + ("\n" if pair_lines else ""))

    if boundary:
        _write_csv(
            options.report_dir / "boundary_candidates.csv",
            ["image_a", "image_b", "component_a", "component_b", "shared_points",
             "distance", "view_angle_deg", "already_matched", "source"],
            candidate_rows,
        )
        diagnostic_only = len(candidate_rows) - len(unmatched)
        if diagnostic_only:
            logger.info(
                "%d boundary pair(s) already have verified matches — re-matching them would "
                "change nothing; they need sharper frames or a looser ratio test instead",
                diagnostic_only,
            )
        logger.warning("Wrote %d candidate bridging pair(s) to %s", len(pair_lines), pair_file)
        logger.warning(
            "  To try bridging (this MUTATES the database — copy it first):\n"
            "    colmap matches_importer --database_path %s \\\n"
            "      --match_list_path %s --match_type pairs --SiftMatching.use_gpu 0",
            options.database_path, pair_file,
        )

    # --- plot ---------------------------------------------------------------
    plot = PlotResult(status="skipped", reason="disabled")
    if options.plot:
        clip_series: dict[str, list[int]] = defaultdict(list)
        for dense, image_id in enumerate(image_ids):
            clip_series[_clip_of(primary.images[image_id].name)].append(dense)
        plot = render_top_down(
            centers,
            [primary.images[i].group or UNGROUPED for i in image_ids],
            labels if len(unique_labels) > 1 else None,
            options.report_dir / "camera_positions_top_down.png",
            clip_series=dict(clip_series),
            dpi=options.plot_dpi,
            title_suffix=f" — {primary_path.name}",
        )
        if plot.status == "written":
            logger.info("Wrote %s", plot.path)

    # --- artifacts + verdict -------------------------------------------------
    _write_csv(
        options.report_dir / "registration_by_group.csv",
        ["group", "total", "registered", "rate"],
        [[r["key"], r["total"], r["registered"], f"{r['rate']:.4f}"] for r in group_rows],
    )

    failed = [c for c in checks if c.severity == "fail" and not c.passed]
    warned = [c for c in checks if c.severity == "warn" and not c.passed]

    report["primary_model"] = {
        "path": primary_path.as_posix(),
        "registered_images": registered,
        "registration_rate": rate,
        "union_registration_rate_informational": union / inventory.total,
        "points": primary.num_points3d,
        "observations": primary.num_observations,
        "reprojection_error": asdict(errors),
        "mean_track_length": primary.mean_track_length(),
        "short_track_fraction": short_tracks,
        "observations_per_image": distribution,
        "bounding_box_extent": extent.tolist(),
        "components": {
            "threshold": options.min_common_points,
            "count": int(len(sizes)),
            "sizes": sorted((int(s) for s in sizes), reverse=True),
            "counted_for_pass_fail": len(big),
            "threshold_curve": [{"min_common_points": t, "components": c} for t, c in curve],
            "skipped_long_tracks": pairs.skipped_long_tracks,
        },
        "registration_by_group": group_rows,
        "registration_by_clip": clip_rows,
        "plot": asdict(plot) | {"path": plot.path.as_posix() if plot.path else None},
    }
    report["checks"] = [asdict(c) for c in checks]
    report["status"] = "fail" if failed else "pass"
    report["exit_code"] = EXIT_FAILED if failed else EXIT_OK

    report_path = options.report_dir / "verify_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    logger.info("=" * 72)
    for check in checks:
        status = "PASS" if check.passed else ("FAIL" if check.severity == "fail" else "WARN")
        log = logger.info if check.passed else (
            logger.error if check.severity == "fail" else logger.warning
        )
        log("  %-4s %-22s %s", status, check.name, check.detail)
    logger.info("=" * 72)

    if failed:
        logger.error("VERIFY FAILED — %d check(s) failed, %d warning(s)", len(failed), len(warned))
        for check in failed:
            logger.error("  %s: %s", check.name, check.detail)
            if check.remediation:
                logger.error("    -> %s", check.remediation)
    else:
        logger.info("VERIFY PASSED — %d warning(s)", len(warned))

    logger.info("Report: %s", report_path)
    return EXIT_FAILED if failed else EXIT_OK
