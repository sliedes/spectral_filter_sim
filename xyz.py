import numpy as np
import numpy.typing as npt
from skimage.color import xyz2rgb

# TODO: This wavelen->XYZ is apparently only an approximation, formulas from Wikipedia


def gauss(x: npt.ArrayLike, mu: float, sigma1: float, sigma2: float) -> npt.NDArray:
    x = np.asarray(x)
    sigma = np.zeros_like(x, dtype=float)
    sigma[x < mu] = sigma1
    sigma[x >= mu] = sigma2
    return np.exp(-0.5 * (x - mu) ** 2 / sigma**2)


def wavelen_to_x(lam: float) -> float:
    return (
        1.056 * gauss(lam, 599.8, 37.9, 31.0)
        + 0.362 * gauss(lam, 442.0, 16.0, 26.7)
        - 0.065 * gauss(lam, 501.1, 20.4, 26.2)
    )


def wavelen_to_y(lam: npt.ArrayLike) -> npt.ArrayLike:
    return 0.821 * gauss(lam, 568.8, 46.9, 40.5) + 0.286 * gauss(lam, 530.9, 16.3, 31.1)


def wavelen_to_z(lam: npt.ArrayLike) -> npt.ArrayLike:
    return 1.217 * gauss(lam, 437.0, 11.8, 36.0) + 0.681 * gauss(lam, 459.0, 26.0, 13.8)


def wavelen_to_xyz(lam: npt.ArrayLike) -> npt.NDArray:
    return np.array([wavelen_to_x(lam), wavelen_to_y(lam), wavelen_to_z(lam)])
