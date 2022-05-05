#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
import cv2
import sys

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

import xyz
import illuminant

WAVELEN_BASE = 400
WAVELEN_INC = 10
NUM_IMAGES = 31

DIRNAME = sys.argv[1].rstrip("/")

WAVELENS = np.arange(WAVELEN_BASE, WAVELEN_BASE + WAVELEN_INC * NUM_IMAGES, WAVELEN_INC)
D65 = np.array([illuminant.D65[x] for x in WAVELENS])

im = (
    np.array(
        [
            cv2.imread(f"data/{DIRNAME}/{DIRNAME}_{chan:02d}.png", cv2.IMREAD_UNCHANGED)
            for chan in range(1, 1 + NUM_IMAGES)
        ]
    ).astype(float)
    / 65535.0
)

xyz_of_wavelen = xyz.wavelen_to_xyz(WAVELENS)


def normalize(a: npt.ArrayLike):
    a = np.asarray(a)
    return a / a.max()


def calc_image(muls):
    a = xyz.xyz2rgb(
        normalize((xyz_of_wavelen * D65 * muls).dot(im.transpose(1, 0, 2)).transpose(1, 2, 0))
    )
    print(a.max())
    a /= a.max()
    return a


image = plt.imshow(calc_image(np.ones((NUM_IMAGES), dtype=float)))

plt.show()
