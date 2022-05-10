#!/usr/bin/env python3

import sys, functools, os
from typing import Union, Tuple

import numpy as np
import numpy.typing as npt
import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.decomposition import PCA

from my_types import FloatArr

try:
    import colored_traceback

    colored_traceback.add_hook(always=True)
except ImportError:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

import lms
import illuminant
from color import linrgb_to_srgb, normalize

TF_FLOAT = tf.float64
NP_FLOAT = np.float64

NpFloatArr = npt.NDArray[NP_FLOAT]

WAVELEN_BASE = 400
WAVELEN_INC = 10
NUM_IMAGES = 31

WAVELEN_RED = 630
WAVELEN_GREEN = 532
WAVELEN_BLUE = 465

# Inverse scale factor for images (for speed). For example, use 2 to resize to 50%.
# For optimizing filters, this is set from OPT_DOWNSCALE.
DOWNSCALE = 1
OPT_DOWNSCALE = 4


IMG_WAVELENS = tf.range(
    WAVELEN_BASE, WAVELEN_BASE + WAVELEN_INC * NUM_IMAGES, WAVELEN_INC, dtype=tf.int32
)  # (NUM_IMAGES,)
PREC_WAVELENS: tf.Tensor = tf.constant(sorted(lms.LMS.keys()), dtype=tf.int32)
D65: tf.Tensor = tf.constant([illuminant.D65[int(x)] for x in IMG_WAVELENS], dtype=TF_FLOAT)  # (NUM_IMAGES,)

STDLMS: tf.Tensor = tf.constant([lms.LMS[x] for x in lms.LMS.keys()], dtype=TF_FLOAT)
LMS_WAVELEN_MIN = min(lms.LMS.keys())
LMS_WAVELEN_MAX = max(lms.LMS.keys())

# Whether to apply D65 when loading
STATIC_ILLUMINATION = True


def lms_at(lms: tf.Tensor, lam_: Union[float, tf.Tensor]) -> tf.Tensor:
    lam = tf.convert_to_tensor(lam_, dtype=tf.int32)
    if tf.rank(lam) == 0:
        assert lam >= LMS_WAVELEN_MIN and lam <= LMS_WAVELEN_MAX, lam
    else:
        assert tf.reduce_all(lam >= LMS_WAVELEN_MIN) and tf.reduce_all(lam <= LMS_WAVELEN_MAX), lam
    return lms[lam - LMS_WAVELEN_MIN]


# While we simulate arbitrary LMS, for displaying to the user in RGB we still want to use standard LMS
# (could make this configurable though)
RGB_TO_STDLMS: tf.Tensor = tf.stack(
    [lms_at(STDLMS, WAVELEN_RED), lms_at(STDLMS, WAVELEN_GREEN), lms_at(STDLMS, WAVELEN_BLUE)],
).T
STDLMS_TO_RGB = tf.linalg.inv(RGB_TO_STDLMS)  # (3, 3)


def sample_lms(lms: tf.Tensor) -> tf.Tensor:  # (NUM_IMAGES, 3) or (441, 3) -> (NUM_IMAGES, 3)
    """Sample the given LMS spectrum at points that correspond to the image bands."""
    # Support the parameter being already sampled
    if lms.shape[-1] == len(IMG_WAVELENS):
        return lms
    return lms_at(lms, IMG_WAVELENS)


def load_gray_img(fname: str) -> FloatArr:  # (height, width)
    """Load image, possibly scaling it."""
    a = cv2.imread(fname, cv2.IMREAD_UNCHANGED).astype(float) / 65535.0
    assert len(a.shape) == 2, a.shape
    if DOWNSCALE != 1:
        a = cv2.resize(a, (a.shape[0] // DOWNSCALE, a.shape[1] // DOWNSCALE), interpolation=cv2.INTER_CUBIC)
    return a  # type: ignore[no-any-return]


@functools.lru_cache(maxsize=5)
def load_multi_image_noall(name: str) -> tf.Tensor:  # (height, width, channel)
    assert os.path.isdir(f"data/{name}"), name
    print(f"Loading image {name}...")
    a = tf.transpose(
        tf.constant(
            [load_gray_img(f"data/{name}/{name}_{chan:02d}.png") for chan in range(1, 1 + NUM_IMAGES)],
            dtype=TF_FLOAT,
        ),
        perm=(1, 2, 0),
    )
    if STATIC_ILLUMINATION:
        a = a * tf.reshape(D65, (1, 1, -1))
    return a


WHOLE_IMAGE: tf.Tensor = None


def load_all_multi_images() -> tf.Tensor:  # (height, width, channel)
    global WHOLE_IMAGE
    if WHOLE_IMAGE is not None:
        return WHOLE_IMAGE
    ims = [load_multi_image_noall.__wrapped__(name) for name in sorted(os.listdir("data/"))]
    WHOLE_IMAGE = tf.concat(ims, 0)
    return WHOLE_IMAGE


def load_multi_image(name: str) -> tf.Tensor:  # (height, width, channel)
    """Load a multichannel image.

    With special name '__all', load all images and stitch them together.
    """
    if name == "__all":
        return load_all_multi_images()
    return load_multi_image_noall(name)


JACOBIAN_EPSILON = 1e-4


def vec_plus_epsilons(vec: tf.Tensor) -> tf.Tensor:  # (n,) -> (n, n)
    assert len(vec.shape) == 1, vec.shape
    a: tf.Tensor = tf.tile(vec, (vec.shape[0], 1))
    tf.fill_diagonal(a, a.diagonal() + JACOBIAN_EPSILON)
    return a


# @timer
def multi_to_lms_adapted(im: tf.Tensor, my_lms: tf.Tensor) -> tf.Tensor:
    # We want to set LMS scale factors so that D65 white maps to monitor native white, i.e. RGB 100%, 100%, 100%.
    # This corresponds to chromatic adaptation to D65 illumination.
    #
    # (D65 * MULTI_TO_LMS) ⊙ x * LMS_TO_RGB^T = 1.T
    # -->
    # x = (1/(D65 * MULTI_TO_LMS)) ⊙ RGB_TO_LMS * 1
    my_lms_sample = sample_lms(my_lms)
    # LMS_FACTORS = (1 / D65.numpy().dot(my_lms_sample.numpy())) * RGB_TO_STDLMS.numpy().dot([1, 1, 1])
    LMS_FACTORS = (1 / tf.tensordot(D65, my_lms_sample, axes=(0, 0))) * tf.math.reduce_sum(RGB_TO_STDLMS, 1)

    illum = my_lms_sample * LMS_FACTORS
    if not STATIC_ILLUMINATION:
        illum = tf.expand_dims(D65, 1) * illum

    return normalize(tf.matmul(im, illum))


# @timer
def load_adapted_image_lms(name: str, my_lms: tf.Tensor) -> tf.Tensor:
    """Load an image in the LMS space with the given modified LMS responses,
    with chromatic adaptation adjustment corresponding to D65.

    With special name '__all', load all images.
    """
    return multi_to_lms_adapted(load_multi_image(name), my_lms)


def std_lms_to_rgb(im: tf.Tensor) -> tf.Tensor:
    """Transfer an LMS image to the RGB space, for an observer with normal responses to trichromatic RGB"""
    return im.dot(STDLMS_TO_RGB.T)


# @timer
def load_adapted_image_linrgb(name: str, my_lms: tf.Tensor) -> tf.Tensor:  # (height, width, 3)
    """Load an image the linear RGB space with the given modified LMS responses,
    with chromatic adaptation adjustment corresponding to D65.

    With special name '__all', load all images.
    """
    return normalize(std_lms_to_rgb(load_adapted_image_lms(name, my_lms)))


def interp_freqs(freqs: tf.Tensor) -> tf.Tensor:
    f = interp1d(IMG_WAVELENS, freqs, kind="linear", bounds_error=False, fill_value=1.0)
    return tf.convert_to_tensor(f(PREC_WAVELENS), dtype=TF_FLOAT)


def shift_lms(lms_: tf.Tensor, l: int, m: int, s: int) -> tf.Tensor:
    lms = np.zeros_like(lms_)
    for i, amount in enumerate([l, m, s]):
        if amount == 0:
            lms[:, i] = lms_[:, i]
        elif amount > 0:
            lms[amount:, i] = lms_[:-amount, i]
        else:
            lms[:amount, i] = lms_[-amount:, i]
    return tf.convert_to_tensor(lms)


def separation_score(lmsimg: NpFloatArr) -> tf.Tensor:
    im = tf.convert_to_tensor(lmsimg.reshape(-1, 3), dtype=TF_FLOAT)
    eigval, _ = tf.linalg.eigh(tf.tensordot(tf.transpose(im), im, axes=1))
    # print("Eigenvalues:", eigval)
    return eigval[0] / tf.reduce_sum(eigval) * 100


# Apparently deuteranomalic M is usually shifted by about 17 nm
# TODO: read literature on whether it still retains the same shape
DEUT_LMS = shift_lms(STDLMS, 0, 17, 0)


def l65_brightness(filt: tf.Tensor, target_lms: tf.Tensor) -> tf.Tensor:
    unfiltered = tf.math.reduce_sum(tf.expand_dims(D65, 1) * sample_lms(target_lms) * lms.TO_LUMINANCE)
    filtered = tf.math.reduce_sum(tf.expand_dims(D65 * filt, 1) * sample_lms(target_lms) * lms.TO_LUMINANCE)
    return filtered / unfiltered


def score_of_filter(filt: tf.Tensor, name: str, target_lms: tf.Tensor) -> float:
    filt = tf.cast(filt, TF_FLOAT)
    lmsimg = load_adapted_image_lms(name, target_lms * interp_freqs(filt).reshape(-1, 1))
    sep_score = separation_score(lmsimg) * 1000
    print("Separation score:", float(sep_score))
    lum = l65_brightness(filt, target_lms)
    # 4/lum^3 seems reasonable
    lum_score = -((1.0 / lum) ** 3) * 2
    print("L:", float(lum))
    print("L score:", float(lum_score))
    score = sep_score + lum_score
    print("Total:", float(score))
    print(filt)
    print()

    return float(-score)


def main() -> None:
    if len(sys.argv) >= 2:
        image_name = sys.argv[1]
    else:
        image_name = "__all"  # "balloons_ms"

    global DOWNSCALE
    DOWNSCALE = OPT_DOWNSCALE

    while True:
        # filt = tf.ones((NUM_IMAGES,), dtype=TF_FLOAT) * 0.6
        filt = tf.convert_to_tensor(np.random.uniform(0.3, 0.9, (NUM_IMAGES,)), dtype=TF_FLOAT)
        res = minimize(
            score_of_filter,
            filt,
            args=(image_name, shift_lms(STDLMS, 0, 0, 0)),
            bounds=((0.01, 1.0),) * NUM_IMAGES,
            jac=False,
        )
        print(res)

        for freq, v in zip(IMG_WAVELENS, res.x):
            print(int(freq), float(v))

        print()
        print()


if __name__ == "__main__":
    main()
