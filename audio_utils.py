import numpy as np
from os import listdir
from os.path import isfile, join
import soundfile as sf


def stereo_to_mono(sample: np.ndarray) -> np.ndarray:
    if sample.ndim == 1:
        res = sample
    else:
        res = np.mean(sample, 1)
    return res


def normalize(sample: np.ndarray) -> np.ndarray:
    """
    Subtract the mean, and scale to the interval [-1,1]
    """
    sample_minusmean = sample - sample.mean()
    return sample_minusmean/abs(sample_minusmean).max()