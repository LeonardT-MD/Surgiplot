from __future__ import annotations

import numpy as np
import pytest

from surgiplot.core.transforms.spaces import apply_affine


def test_apply_affine_transforms_point_matrix() -> None:
    affine = np.eye(4)
    affine[:3, 3] = [10.0, -20.0, 30.0]
    transformed = apply_affine([[1, 2, 3], [4, 5, 6]], affine)
    np.testing.assert_allclose(transformed, [[11, -18, 33], [14, -15, 36]])


@pytest.mark.parametrize(
    ("points", "affine"),
    [
        ([[1, 2]], np.eye(4)),
        ([[1, 2, np.nan]], np.eye(4)),
        ([[1, 2, 3]], np.eye(3)),
    ],
)
def test_apply_affine_rejects_invalid_inputs(points, affine) -> None:
    with pytest.raises(ValueError):
        apply_affine(points, affine)
