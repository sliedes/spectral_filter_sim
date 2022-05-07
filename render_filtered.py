#!/usr/bin/env python3

import sys
import functools

import numpy as np
import numpy.typing as npt
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

import lms
import illuminant
from color import srgb_to_linrgb, linrgb_to_srgb, normalize

WAVELEN_BASE = 400
WAVELEN_INC = 10
NUM_IMAGES = 31

WAVELEN_RED = 630
WAVELEN_GREEN = 532
WAVELEN_BLUE = 465

WAVELENS = np.arange(WAVELEN_BASE, WAVELEN_BASE + WAVELEN_INC * NUM_IMAGES, WAVELEN_INC)  # (NUM_IMAGES,)
D65: npt.NDArray[np.float_] = np.array([illuminant.D65[x] for x in WAVELENS])  # (NUM_IMAGES,)
MULTI_TO_LMS: npt.NDArray[np.float_] = np.array([lms.LMS[x] for x in WAVELENS])  # (NUM_IMAGES, 3)


RGB_TO_LMS: npt.NDArray[np.float_] = np.array(
    [lms.LMS[WAVELEN_RED], lms.LMS[WAVELEN_GREEN], lms.LMS[WAVELEN_BLUE]]
).T  # (3, 3)

LMS_TO_RGB = np.linalg.inv(RGB_TO_LMS)  # (3, 3)

# We want to set LMS scale factors so that D65 white maps to monitor native white, i.e. RGB 100%, 100%, 100%.
# This corresponds to chromatic adaptation to D65 illumination.
#
# (D65 * MULTI_TO_LMS) ⊙ x * LMS_TO_RGB^T = 1.T
# -->
# x = (1/(D65 * MULTI_TO_LMS)) ⊙ RGB_TO_LMS * 1
LMS_FACTORS = (1 / D65.dot(MULTI_TO_LMS)) * RGB_TO_LMS.dot([1, 1, 1])


@functools.lru_cache(maxsize=5)
def load_image(name: str) -> npt.NDArray[np.float_]:  # (height, width, channel)
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


def load_lms_image(name: str, muls: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:  # (height, width, 3)
    """Returns an image in the LMS space with chromatic adaptation adjustment corresponding to D65."""
    im = load_image(name)  # (height, width, NUM_IMAGES)
    return normalize((im * D65 * muls).dot(MULTI_TO_LMS) * LMS_FACTORS)


def main() -> None:
    if len(sys.argv) >= 2:
        image_name = sys.argv[1]
    else:
        image_name = "balloons_ms"
    # plt.imshow(calc_image(image_name, np.ones((NUM_IMAGES), dtype=float)))
    lms = load_lms_image(image_name, np.ones((NUM_IMAGES), dtype=float))
    linrgb = normalize(lms.dot(LMS_TO_RGB.T))
    srgb = linrgb_to_srgb(linrgb)
    # plt.imsave("rendered.png", srgb)
    plt.imshow(srgb)
    plt.show()


if __name__ == "__main__":
    main()
