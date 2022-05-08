#!/usr/bin/env python3

import sys, functools, os, threading

import numpy as np
import numpy.typing as npt
import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.decomposition import PCA

import lms
import illuminant
from color import linrgb_to_srgb, normalize
from my_types import FloatArr, Int32Arr

WAVELEN_BASE = 400
WAVELEN_INC = 10
NUM_IMAGES = 31

WAVELEN_RED = 630
WAVELEN_GREEN = 532
WAVELEN_BLUE = 465

IMG_WAVELENS = np.arange(WAVELEN_BASE, WAVELEN_BASE + WAVELEN_INC * NUM_IMAGES, WAVELEN_INC)  # (NUM_IMAGES,)
PREC_WAVELENS: FloatArr = np.array(sorted(lms.LMS.keys()), dtype=np.float32)
D65: FloatArr = np.array([illuminant.D65[x] for x in IMG_WAVELENS])  # (NUM_IMAGES,)

LMS_STD: FloatArr = np.array([lms.LMS[x] for x in lms.LMS.keys()], dtype=np.float32)
LMS_WAVELEN_MIN = min(lms.LMS.keys())
LMS_WAVELEN_MAX = max(lms.LMS.keys())


def lms_at(lms: FloatArr, lam_: npt.ArrayLike) -> FloatArr:
    lam: Int32Arr = np.asarray(lam_, dtype=np.int32)
    assert np.all(lam >= LMS_WAVELEN_MIN) and np.all(lam <= LMS_WAVELEN_MAX), lam
    return lms[lam - LMS_WAVELEN_MIN]  # type: ignore[no-any-return]


# While we simulate arbitrary LMS, for displaying to the user in RGB we still want to use standard LMS
# (could make this configurable though)
RGB_TO_STD_LMS: FloatArr = np.array(
    [lms_at(LMS_STD, WAVELEN_RED), lms_at(LMS_STD, WAVELEN_GREEN), lms_at(LMS_STD, WAVELEN_BLUE)], dtype=np.float32
).T
STD_LMS_TO_RGB = np.linalg.inv(RGB_TO_STD_LMS)  # (3, 3)


def sample_lms(lms: FloatArr) -> FloatArr:
    """Sample the given LMS spectrum at points that correspond to the image bands."""
    # Support the parameter being already sampled
    if lms.shape[-1] == len(IMG_WAVELENS):
        return lms
    return lms_at(lms, IMG_WAVELENS)


@functools.lru_cache(maxsize=5)
def load_multi_image_noall(name: str) -> FloatArr:  # (height, width, channel)
    assert os.path.isdir(f"data/{name}"), name
    a = (
        np.array(
            [
                cv2.imread(f"data/{name}/{name}_{chan:02d}.png", cv2.IMREAD_UNCHANGED)
                for chan in range(1, 1 + NUM_IMAGES)
            ]
        )
        .astype(np.float32)
        .transpose(1, 2, 0)
        / 65535.0
    )
    return a


WHOLE_IMAGE: FloatArr = None  # type: ignore[assignment]


def load_all_multi_images() -> FloatArr:  # (height, width, channel)
    global WHOLE_IMAGE
    if WHOLE_IMAGE is not None:
        return WHOLE_IMAGE
    ims = [load_multi_image_noall.__wrapped__(name) for name in sorted(os.listdir("data/"))]
    WHOLE_IMAGE = np.concatenate(ims)
    return WHOLE_IMAGE


def load_multi_image(name: str) -> FloatArr:  # (height, width, channel)
    """Load a multichannel image.

    With special name '__all', load all images and stitch them together.
    """
    if name == "__all":
        return load_all_multi_images()
    return load_multi_image_noall(name)


JACOBIAN_EPSILON = 1e-4


def vec_plus_epsilons(vec: FloatArr) -> FloatArr:  # (n,) -> (n, n)
    assert len(vec.shape) == 1, vec.shape
    a: FloatArr = np.tile(vec, (vec.shape[0], 1))
    np.fill_diagonal(a, a.diagonal() + JACOBIAN_EPSILON)
    return a


def multi_to_lms_adapted(im: FloatArr, my_lms: FloatArr) -> FloatArr:
    # We want to set LMS scale factors so that D65 white maps to monitor native white, i.e. RGB 100%, 100%, 100%.
    # This corresponds to chromatic adaptation to D65 illumination.
    #
    # (D65 * MULTI_TO_LMS) ⊙ x * LMS_TO_RGB^T = 1.T
    # -->
    # x = (1/(D65 * MULTI_TO_LMS)) ⊙ RGB_TO_LMS * 1
    # LMS_FACTORS = (1 / D65.dot(my_lms_sample)) * RGB_TO_LMS.dot([1, 1, 1])
    my_lms_sample = sample_lms(my_lms)
    LMS_FACTORS = (1 / D65.dot(my_lms_sample)) * RGB_TO_STD_LMS.dot([1, 1, 1])
    return normalize((im * D65).dot(my_lms_sample) * LMS_FACTORS)


def load_adapted_image_lms(name: str, my_lms: FloatArr) -> FloatArr:
    """Load an image in the LMS space with the given modified LMS responses,
    with chromatic adaptation adjustment corresponding to D65.

    With special name '__all', load all images.
    """
    return multi_to_lms_adapted(load_multi_image(name), my_lms)


def std_lms_to_rgb(im: FloatArr) -> FloatArr:
    """Transfer an LMS image to the RGB space, for an observer with normal responses to trichromatic RGB"""
    return im.dot(STD_LMS_TO_RGB.T)


def load_adapted_image_linrgb(name: str, my_lms: FloatArr) -> FloatArr:  # (height, width, 3)
    """Load an image the linear RGB space with the given modified LMS responses,
    with chromatic adaptation adjustment corresponding to D65.

    With special name '__all', load all images.
    """
    return normalize(std_lms_to_rgb(load_adapted_image_lms(name, my_lms)))


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


def separation_score(lmsimg: FloatArr) -> float:
    pca = PCA()
    lmsimg = lmsimg.reshape(-1, 3)
    pca.fit(lmsimg)
    return pca.explained_variance_ratio_[-1] * 100  # type: ignore[no-any-return]


# Apparently deuteranomalic M is usually shifted by about 17 nm
# TODO: read literature on whether it still retains the same shape
DEUT_LMS = shift_lms(LMS_STD, 0, 17, 0)


def l65_brightness(filt: FloatArr, target_lms: FloatArr) -> float:
    unfiltered = (D65).dot(sample_lms(target_lms)).sum()
    filtered = (D65 * filt).dot(sample_lms(target_lms)).sum()
    return filtered / unfiltered  # type: ignore[no-any-return]


def score_of_filter(filt: FloatArr, name: str, target_lms: FloatArr) -> float:
    lmsimg = load_adapted_image_lms(name, target_lms * interp_freqs(filt).reshape(-1, 1))
    score = separation_score(lmsimg)
    print("Score:", score)
    print("L:", l65_brightness(filt, target_lms))
    print(filt)
    print()
    return -separation_score(lmsimg)


def main() -> None:
    if len(sys.argv) >= 2:
        image_name = sys.argv[1]
    else:
        image_name = "__all"  # "balloons_ms"

    filt = np.ones((NUM_IMAGES,)) * 0.6
    res = minimize(
        score_of_filter,
        filt,
        args=(image_name, shift_lms(LMS_STD, 0, 0, 0)),
        bounds=((0.2, 1.0),) * NUM_IMAGES,
        options=dict(eps=0.0001),
    )
    print(res)

    for freq, v in zip(IMG_WAVELENS, res.x):
        print(freq, v)


if __name__ == "__main__":
    main()
