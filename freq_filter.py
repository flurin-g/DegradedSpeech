"""
The functions butter_bandpass and butter_bandpass_filter are
taken from: https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html?highlight=bandpass
"""
from random import uniform

import soundfile as sf
from scipy.signal import butter, lfilter

from audio_utils import normalize


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


class FreqFilter:
    def __init__(self, cfg, cwd):
        self.cfg = cfg
        self.cwd = cwd
        self.idx = 0

    def calc_params(self) -> tuple:
        lowcut = int(uniform(self.cfg.hp_low, self.cfg.hp_high))
        highcut = lowcut + self.cfg.bandwidth
        order = self.cfg.order

        return lowcut, highcut, order

    def __call__(self, src: str) -> tuple:
        speech, sr = sf.read(src)
        lowcut, highcut, order = self.calc_params()
        filtered_speech = butter_bandpass_filter(speech, lowcut, highcut, sr, order)
        filtered_normalized = normalize(filtered_speech)
        return filtered_normalized, sr
