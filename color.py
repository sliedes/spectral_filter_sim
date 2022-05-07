import numpy as np
import numpy.typing as npt

from my_types import FloatArr


def normalize(a: FloatArr) -> FloatArr:
    a = np.asarray(a)
    return (a / a.max()).clip(0.0, 1.0)  # type: ignore[no-any-return]


def srgb_to_linrgb(a: FloatArr) -> FloatArr:
    assert a.max() <= 1.0, a.max()
    # This is almost, but not exactly, correct approximation of the sRGB transfer function
    a = a.copy()
    lo_mask = a < 0.0404482362771082
    a[lo_mask] /= 12.92
    a[~lo_mask] = ((a[~lo_mask] + (0.055)) / 1.055) ** 2.4
    return a


def linrgb_to_srgb(a: FloatArr) -> FloatArr:
    assert a.max() <= 1.0, a.max()
    a = a.copy()
    lo_mask = a < 0.00313066844250063
    a[lo_mask] *= 12.92
    a[~lo_mask] = 1.055 * a[~lo_mask] ** (1.0 / 2.4) - 0.055
    return a
