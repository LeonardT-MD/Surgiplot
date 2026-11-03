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


def test_vom_reports_standard_distance_clipping() -> None:
    result = VOM_VOA(entry=_square(10.0), target=_square(0.0), stand_dist=20.0, return_debug=True)
    assert result.svom_mm3 == pytest.approx(result.vom_mm3)
    assert result.debug is not None
    assert result.debug["requested_stand_dist_mm"] == pytest.approx(20.0)
    assert result.debug["stand_dist_mm"] == pytest.approx(10.0)
    assert result.debug["stand_dist_clipped_to_corridor"] is True


def test_vom_rejects_degenerate_polygon() -> None:
    with pytest.raises(ValueError):
        VOM_VOA(
            entry=[[0, 0, 1], [1, 0, 1], [2, 0, 1]],
            target=_square(0.0),
        )
