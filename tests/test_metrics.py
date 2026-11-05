from __future__ import annotations

import numpy as np
import pytest

from surgiplot import AOA_SF, AOE, AREA_3D, DISTANCE_3D, VOLUME_3D, VOM_VOA


def _square(z: float, half_width: float = 1.0) -> np.ndarray:
    return np.asarray(
        [
            [-half_width, -half_width, z],
            [half_width, -half_width, z],
            [half_width, half_width, z],
            [-half_width, half_width, z],
        ],
        dtype=float,
    )


def test_distance_known_value() -> None:
    result = DISTANCE_3D(A=[0, 0, 0], B=[3, 4, 12])
    assert result.distance_mm == pytest.approx(13.0)


def test_area_is_rigid_transform_invariant() -> None:
    polygon = _square(0.0, half_width=2.0)
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = polygon @ rotation.T + np.asarray([1e6, -2e6, 3e6])
    assert AREA_3D(polygon=polygon).area_mm2 == pytest.approx(16.0)
    assert AREA_3D(polygon=transformed).area_mm2 == pytest.approx(16.0)


@pytest.mark.parametrize(
    "polygon",
    [
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        [[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]],
    ],
)
def test_area_rejects_degenerate_polygons(polygon) -> None:
    with pytest.raises(ValueError):
        AREA_3D(polygon=polygon)


def test_volume_of_unit_tetrahedron() -> None:
    result = VOLUME_3D(points=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert result.volume_mm3 == pytest.approx(1.0 / 6.0)


def test_sparse_volume_reconstruction_is_translation_invariant() -> None:
    rng = np.random.default_rng(123)
    points = rng.normal(size=(30, 3))
    baseline = VOLUME_3D(points=points, return_debug=True)
    translated = VOLUME_3D(
        points=points + np.asarray([1e6, -2e6, 3e6]),
        return_debug=True,
    )

    assert baseline.volume_mm3 == pytest.approx(translated.volume_mm3, rel=1e-9, abs=1e-9)
    assert baseline.debug is not None
    assert translated.debug is not None
    assert baseline.debug["reconstruction_method"] == translated.debug["reconstruction_method"]
    assert baseline.debug["alpha_mm"] == pytest.approx(translated.debug["alpha_mm"], rel=1e-9, abs=1e-9)


def test_volume_rejects_coplanar_input() -> None:
    with pytest.raises(ValueError, match="coplanar"):
        VOLUME_3D(points=_square(0.0))


def test_aoe_is_principal_included_angle() -> None:
    result = AOE(A=[1, 0, 0], B=[0, 0, 0], C=[0, 1, 0])
    assert result.aoe_deg == pytest.approx(90.0)


def test_aoe_rejects_zero_length_arm() -> None:
    with pytest.raises(ValueError, match="distinct"):
        AOE(A=[0, 0, 0], B=[0, 0, 0], C=[0, 1, 0])


def test_aoa_sf_cardinal_geometry_and_standardization() -> None:
    entry = np.asarray(
        [
            [0, 1, 0],   # cranial
            [0, -1, 0],  # caudal
            [-1, 0, 0],  # medial
            [1, 0, 0],   # lateral
        ],
        dtype=float,
    )
    result = AOA_SF(entry=entry, target=[0, 0, 1], sf_rescale_radius_mm=1.0)
    assert result.aoa_vertical_deg == pytest.approx(90.0)
    assert result.aoa_horizontal_deg == pytest.approx(90.0)
    assert result.sf_entry_area_mm2 == pytest.approx(2.0)
    assert result.sf_entry_area_rescaled_mm2 == pytest.approx(1.0)


def test_aoa_sf_requires_exact_cardinal_roles() -> None:
    with pytest.raises(ValueError, match="exactly four"):
        AOA_SF(entry=[[0, 0, 0], [1, 0, 0]], target=[0, 1, 0])


def test_aoa_sf_rejects_pivot_equal_to_entry() -> None:
    entry = [[0, 1, 0], [0, -1, 0], [-1, 0, 0], [1, 0, 0]]
    with pytest.raises(ValueError, match="pivot"):
        AOA_SF(entry=entry, target=[0, 1, 0])


def test_vom_voa_known_parallel_corridor_and_translation_invariance() -> None:
    entry = _square(10.0)
    target = _square(0.0)
    baseline = VOM_VOA(entry=entry, target=target, stand_dist=5.0, return_debug=True)
    translated = VOM_VOA(
        entry=entry + np.asarray([1e6, -2e6, 3e6]),
        target=target + np.asarray([1e6, -2e6, 3e6]),
        stand_dist=5.0,
    )

    assert baseline.voa_deg == pytest.approx(90.0, abs=1e-10)
    assert baseline.vom_mm3 == pytest.approx(40.0, rel=5e-4)
    assert baseline.svom_mm3 == pytest.approx(20.0, rel=5e-4)
    assert translated.voa_deg == pytest.approx(baseline.voa_deg, abs=1e-10)
    assert translated.vom_mm3 == pytest.approx(baseline.vom_mm3, rel=1e-10, abs=1e-9)
    assert translated.svom_mm3 == pytest.approx(baseline.svom_mm3, rel=1e-10, abs=1e-9)
    assert baseline.debug is not None
    assert baseline.debug["method_version"] == "3d-loft-2.0"
    assert baseline.debug["ellipse_polygon_max_relative_area_error"] == pytest.approx(1e-4)


def test_vom_reports_standard_distance_clipping() -> None:
    result = VOM_VOA(entry=_square(10.0), target=_square(0.0), stand_dist=20.0, return_debug=True)
    assert result.svom_mm3 == pytest.approx(result.vom_mm3)
    assert result.debug is not None
    assert result.debug["requested_stand_dist_mm"] == pytest.approx(20.0)
    assert result.debug["stand_dist_mm"] == pytest.approx(10.0)
    assert result.debug["stand_dist_clipped_to_corridor"] is True
    assert result.debug["svom_longitudinal_convergence"]["reused_full_vom_mesh"] is True


def test_voa_convention_is_ideal_perpendicular_and_worst_tangential() -> None:
    perpendicular = VOM_VOA(entry=_square(10.0), target=_square(0.0))
    tangential = VOM_VOA(entry=_square(0.0) + np.asarray([10.0, 0.0, 0.0]), target=_square(0.0))
    assert perpendicular.voa_deg == pytest.approx(90.0, abs=1e-10)
    assert tangential.voa_deg == pytest.approx(0.0, abs=1e-10)


def test_vom_direct_ellipse_fit_rejects_single_contour_outlier() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    target = np.column_stack([2.0 * np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    entry = target + np.asarray([0.0, 0.0, 10.0])
    contaminated = target.copy()
    contaminated[3, :2] = [20.0, 20.0]

    result = VOM_VOA(entry=entry, target=contaminated, return_debug=True)
    assert result.debug is not None
    assert result.debug["target_ellipse_fit_method"] == "direct_least_squares_tukey_irls"
    assert result.debug["target_ellipse_fit_outlier_indices"] == [3]
    assert result.debug["target_polygon_area_mm2"] == pytest.approx(6.0646748343, rel=1e-9)
    assert result.debug["target_raw_polygon_area_mm2"] > 20.0


def test_vom_is_invariant_to_contour_start_and_direction_after_frame_alignment() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    target = np.column_stack([3.0 * np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    angle = np.radians(55.0)
    rotation = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]]
    )
    entry = target @ rotation.T + np.asarray([1.0, 2.0, 10.0])

    baseline = VOM_VOA(entry=entry, target=target, stand_dist=5.0, return_debug=True)
    reordered = VOM_VOA(entry=entry[::-1], target=np.roll(target, 5, axis=0), stand_dist=5.0, return_debug=True)
    assert reordered.voa_deg == pytest.approx(baseline.voa_deg, abs=1e-10)
    assert reordered.vom_mm3 == pytest.approx(baseline.vom_mm3, rel=1e-10, abs=1e-9)
    assert reordered.svom_mm3 == pytest.approx(baseline.svom_mm3, rel=1e-10, abs=1e-9)
    assert baseline.debug is not None
    assert baseline.debug["orientation_interpolation"] == "aligned_quaternion_slerp"
    assert baseline.debug["target_normal_method"] == "normalized_svd_plane_normal"
    assert baseline.debug["vom_longitudinal_convergence"]["converged"] is True


def test_vom_uses_minimum_rotation_axis_swap_for_large_in_plane_rotation() -> None:
    """A +70° major-axis fit is the same ellipse as a −20° swapped frame."""
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    target = np.column_stack([4.0 * np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    entry_local = np.column_stack([5.0 * np.cos(theta), 1.5 * np.sin(theta)])
    angle = np.radians(70.0)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    entry_xy = entry_local @ rotation.T
    entry = np.column_stack([entry_xy, np.full(theta.shape, 100.0)])

    result = VOM_VOA(entry=entry, target=target, stand_dist=50.0, return_debug=True)
    debug = result.debug or {}

    assert debug["ellipse_frame_correspondence_policy"] == "minimum_rotation_with_axis_permutation"
    assert debug["entry_frame_alignment_swapped_axes"] is True
    assert str(debug["entry_frame_alignment_operation"]).startswith("swap_")
    assert debug["entry_frame_alignment_rotation_deg"] == pytest.approx(20.0, abs=1e-8)
    assert debug["entry_ellipse_a_mm"] < debug["entry_ellipse_b_mm"]

    a0 = float(debug["target_ellipse_a_mm"])
    b0 = float(debug["target_ellipse_b_mm"])
    a1 = float(debug["entry_ellipse_a_mm"])
    b1 = float(debug["entry_ellipse_b_mm"])
    da = a1 - a0
    db = b1 - b0
    expected_option_b = np.pi * 100.0 * (a0 * b0 + 0.5 * (a0 * db + b0 * da) + da * db / 3.0)

    historical_da = b1 - a0
    historical_db = a1 - b0
    historical_major_to_major = np.pi * 100.0 * (
        a0 * b0
        + 0.5 * (a0 * historical_db + b0 * historical_da)
        + historical_da * historical_db / 3.0
    )
    assert result.vom_mm3 == pytest.approx(expected_option_b, rel=5e-4)
    assert result.vom_mm3 > historical_major_to_major * 1.25


def test_vom_rejects_degenerate_polygon() -> None:
    with pytest.raises(ValueError):
        VOM_VOA(
            entry=[[0, 0, 1], [1, 0, 1], [2, 0, 1]],
            target=_square(0.0),
        )
