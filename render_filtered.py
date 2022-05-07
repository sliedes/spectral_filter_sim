#!/usr/bin/env python3

import sys
import functools

import numpy as np
import numpy.typing as npt
import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

import lms
import illuminant
from color import linrgb_to_srgb, normalize
from my_types import FloatArr

WAVELEN_BASE = 400
WAVELEN_INC = 10
NUM_IMAGES = 31

WAVELEN_RED = 630
WAVELEN_GREEN = 532
WAVELEN_BLUE = 465

IMG_WAVELENS = np.arange(WAVELEN_BASE, WAVELEN_BASE + WAVELEN_INC * NUM_IMAGES, WAVELEN_INC)  # (NUM_IMAGES,)
PREC_WAVELENS: FloatArr = np.array(sorted(lms.LMS.keys()))
D65: FloatArr = np.array([illuminant.D65[x] for x in IMG_WAVELENS])  # (NUM_IMAGES,)

LMS_STD: npt.NDArray[np.float_] = np.array([lms.LMS[x] for x in lms.LMS.keys()])
LMS_WAVELEN_MIN = min(lms.LMS.keys())
LMS_WAVELEN_MAX = max(lms.LMS.keys())


def lms_at(lms: FloatArr, lam_: npt.ArrayLike) -> FloatArr:
    lam: npt.NDArray[np.int_] = np.asarray(lam_)
    assert np.all(lam >= LMS_WAVELEN_MIN) and np.all(lam <= LMS_WAVELEN_MAX), lam
    return lms[lam - LMS_WAVELEN_MIN]  # type: ignore[no-any-return]


# While we simulate arbitrary LMS, for displaying to the user in RGB we still want to use standard LMS
# (could make this configurable though)
RGB_TO_STD_LMS: FloatArr = np.array(
    [lms_at(LMS_STD, WAVELEN_RED), lms_at(LMS_STD, WAVELEN_GREEN), lms_at(LMS_STD, WAVELEN_BLUE)]
).T
STD_LMS_TO_RGB = np.linalg.inv(RGB_TO_STD_LMS)  # (3, 3)


def sample_lms(lms: FloatArr) -> FloatArr:
    """Sample the given LMS spectrum at points that correspond to the image bands."""
    return lms_at(lms, IMG_WAVELENS)


@functools.lru_cache(maxsize=5)
def load_multi_image(name: str) -> FloatArr:  # (height, width, channel)
    a = (
        np.array(
            [
                cv2.imread(f"data/{name}/{name}_{chan:02d}.png", cv2.IMREAD_UNCHANGED)
                for chan in range(1, 1 + NUM_IMAGES)
            ]
        )
        .astype(float)
        .transpose(1, 2, 0)
        / 65535.0
    )
    return a


def load_lms_image(name: str, my_lms: FloatArr) -> FloatArr:  # (height, width, 3)
    """Returns an image in the linear RGB space with the given modified LMS responses,
    with chromatic adaptation adjustment corresponding to D65."""
    im = load_multi_image(name)  # (height, width, NUM_IMAGES)

    my_lms_sample = sample_lms(my_lms)

    # RGB_TO_LMS: FloatArr = np.array(
    #     [lms_at(my_lms, WAVELEN_RED), lms_at(my_lms, WAVELEN_GREEN), lms_at(my_lms, WAVELEN_BLUE)]
    # ).T  # (3, 3)

    # We want to set LMS scale factors so that D65 white maps to monitor native white, i.e. RGB 100%, 100%, 100%.
    # This corresponds to chromatic adaptation to D65 illumination.
    #
    # (D65 * MULTI_TO_LMS) ⊙ x * LMS_TO_RGB^T = 1.T
    # -->
    # x = (1/(D65 * MULTI_TO_LMS)) ⊙ RGB_TO_LMS * 1
    # LMS_FACTORS = (1 / D65.dot(my_lms_sample)) * RGB_TO_LMS.dot([1, 1, 1])
    LMS_FACTORS = (1 / D65.dot(my_lms_sample)) * RGB_TO_STD_LMS.dot([1, 1, 1])

    im_lms = normalize((im * D65).dot(my_lms_sample) * LMS_FACTORS)
    return normalize(im_lms.dot(STD_LMS_TO_RGB.T))


def interp_freqs(freqs: FloatArr) -> FloatArr:
    f = interp1d(IMG_WAVELENS, freqs, kind="linear", bounds_error=False, fill_value=1.0)
    return f(PREC_WAVELENS)  # type: ignore[no-any-return]


def shift_lms(lms_: FloatArr, l: int, m: int, s: int) -> FloatArr:
    lms = np.zeros_like(lms_)
    for i, amount in enumerate([l, m, s]):
        if amount == 0:
            lms[:, i] = lms_[:, i]
        elif amount > 0:
            lms[amount:, i] = lms_[:-amount, i]
        else:
            lms[:amount, i] = lms_[-amount:, i]
    return lms


def main() -> None:
    if len(sys.argv) >= 2:
        image_name = sys.argv[1]
    else:
        image_name = "balloons_ms"

    my_lms = LMS_STD.copy()
    # shift M towards higher wavelengths
    my_lms[2:, 1] = my_lms[:-2, 1]

    linrgb = load_lms_image(image_name, my_lms)
    srgb = linrgb_to_srgb(linrgb)
    plt.imsave("rendered.png", srgb)
    plt.imshow(srgb)
    plt.show()


if __name__ == "__main__":
    main()
