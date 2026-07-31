import numpy as np

from pipeline.connectivity import (
    component_threshold_curve,
    connected_components,
    count_shared_points,
    cross_component_pairs,
    spatial_candidates,
)


def build_tracks(tracks):
    """Turn a list of image-id lists into the flat CSR arrays the counter takes."""
    flat = [image_id for track in tracks for image_id in track]
    offsets = np.zeros(len(tracks) + 1, dtype=np.int64)
    np.cumsum([len(t) for t in tracks], out=offsets[1:])
    return np.array(flat, dtype=np.uint32), offsets


def test_counts_shared_points_per_pair():
    image_ids = np.array([1, 2, 3])
    flat, offsets = build_tracks([[1, 2], [1, 2], [2, 3]])

    pairs = count_shared_points(image_ids, flat, offsets)

    counts = {
        (int(a), int(b)): int(c)
        for a, b, c in zip(pairs.pair_lo, pairs.pair_hi, pairs.counts, strict=True)
    }
    assert counts == {(0, 1): 2, (1, 2): 1}


def test_duplicate_image_within_a_track_does_not_self_pair():
    # A point can legitimately land on the same image twice after a bad match.
    image_ids = np.array([1, 2])
    flat, offsets = build_tracks([[1, 1, 2]])

    pairs = count_shared_points(image_ids, flat, offsets)

    assert pairs.num_pairs == 1
    assert (int(pairs.pair_lo[0]), int(pairs.pair_hi[0])) == (0, 1)
    assert int(pairs.counts[0]) == 1


def test_long_tracks_are_skipped_for_pair_counting():
    image_ids = np.arange(1, 11)
    flat, offsets = build_tracks([list(range(1, 11)), [1, 2]])

    pairs = count_shared_points(image_ids, flat, offsets, max_track_length=5)

    assert pairs.skipped_long_tracks == 1
    assert pairs.longest_track == 10
    # Only the short track survives.
    assert pairs.num_pairs == 1
    assert int(pairs.counts[0]) == 1


def test_two_components_are_detected():
    image_ids = np.array([1, 2, 3, 4])
    flat, offsets = build_tracks([[1, 2]] * 5 + [[3, 4]] * 5)

    pairs = count_shared_points(image_ids, flat, offsets)
    labels = connected_components(pairs, min_common_points=5)

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert len(set(labels.tolist())) == 2


def test_threshold_controls_whether_a_weak_link_connects():
    image_ids = np.array([1, 2, 3, 4])
    # Strong links inside each half, a single weak link across the middle.
    tracks = [[1, 2]] * 10 + [[3, 4]] * 10 + [[2, 3]] * 3
    flat, offsets = build_tracks(tracks)
    pairs = count_shared_points(image_ids, flat, offsets)

    assert len(set(connected_components(pairs, 3).tolist())) == 1
    assert len(set(connected_components(pairs, 5).tolist())) == 2


def test_component_zero_is_always_the_largest():
    image_ids = np.array([1, 2, 3, 4, 5])
    flat, offsets = build_tracks([[1, 2]] * 5 + [[2, 3]] * 5 + [[4, 5]] * 5)

    labels = connected_components(count_shared_points(image_ids, flat, offsets), 5)

    _, sizes = np.unique(labels, return_counts=True)
    assert sizes[0] == 3
    assert labels[0] == labels[1] == labels[2] == 0


def test_isolated_image_forms_its_own_component():
    image_ids = np.array([1, 2, 3])
    flat, offsets = build_tracks([[1, 2]] * 5)

    labels = connected_components(count_shared_points(image_ids, flat, offsets), 5)

    assert len(set(labels.tolist())) == 2


def test_threshold_curve_reports_more_components_as_threshold_rises():
    image_ids = np.array([1, 2, 3, 4])
    flat, offsets = build_tracks([[1, 2]] * 10 + [[3, 4]] * 10 + [[2, 3]] * 4)

    curve = component_threshold_curve(count_shared_points(image_ids, flat, offsets), (2, 5, 20))

    assert curve == [(2, 1), (5, 2), (20, 4)]


def test_cross_component_pairs_rank_near_misses_first():
    image_ids = np.array([1, 2, 3, 4])
    # Two clusters; image 2<->3 share 4 points and 1<->4 share 1.
    tracks = [[1, 2]] * 10 + [[3, 4]] * 10 + [[2, 3]] * 4 + [[1, 4]]
    flat, offsets = build_tracks(tracks)
    pairs = count_shared_points(image_ids, flat, offsets)
    labels = connected_components(pairs, min_common_points=6)

    boundary = cross_component_pairs(pairs, labels, min_common_points=6)

    assert [b.shared_points for b in boundary] == [4, 1]
    assert all(b.source == "shared_points" for b in boundary)
    assert all(labels[b.dense_a] != labels[b.dense_b] for b in boundary)


def test_cross_component_pairs_empty_when_single_component():
    image_ids = np.array([1, 2])
    flat, offsets = build_tracks([[1, 2]] * 10)
    pairs = count_shared_points(image_ids, flat, offsets)

    labels = connected_components(pairs, 5)

    assert cross_component_pairs(pairs, labels, 5) == []


def test_spatial_candidates_prefer_nearest_cameras():
    centers = np.array([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0], [50.0, 0, 0]])
    directions = np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))
    labels = np.array([0, 0, 1, 1])

    candidates = spatial_candidates(
        centers, directions, labels, 0, 1, neighbors_per_image=1
    )

    assert candidates
    nearest = candidates[0]
    assert {nearest.dense_a, nearest.dense_b} == {1, 2}
    assert nearest.source == "spatial"
    assert nearest.distance == 9.0


def test_spatial_candidates_reject_cameras_facing_away():
    # Adjacent rooms: cameras are close together but point in opposite
    # directions, i.e. through a wall. These must not be proposed.
    centers = np.array([[0.0, 0, 0], [0.3, 0, 0]])
    directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    labels = np.array([0, 1])

    candidates = spatial_candidates(
        centers, directions, labels, 0, 1, neighbors_per_image=1, max_view_angle_deg=90.0
    )

    assert candidates == []


def test_empty_model_yields_no_pairs():
    pairs = count_shared_points(np.array([]), np.array([], dtype=np.uint32), np.array([0]))

    assert pairs.num_pairs == 0
    assert connected_components(pairs, 5).size == 0
