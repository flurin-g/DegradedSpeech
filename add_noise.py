from random import uniform

import numpy as np
import pandas as pd
import soundfile as sf
from librosa import resample

from audio_utils import stereo_to_mono, normalize


def create_match_to_speech(noise_sr: int, speech_sr: int) -> callable:

    def match_to_speech(sample: np.ndarray) -> np.ndarray:
        """
        Resamples samples to the sampling rate of speech-sample
        additionally converts sample to mono
        @param sample: sample from the urban8k data-set
        @return: transformed sample
        """
        sample_mono = stereo_to_mono(sample)
        return resample(y=sample_mono, orig_sr=noise_sr, target_sr=speech_sr)

    return match_to_speech


def match_length(sample: np.ndarray, length: int) -> np.ndarray:
    """
    Trims an audio-sample to the desired length, if the sample is shorter
    than the desired length, it will be repeated n times and then trimmed
    @param sample: audio-sample to be trimmed
    @param length: desired length
    @return: trimmed audio-sample
    """
    if sample.shape[0] < length:
        times = np.math.ceil(length / sample.shape[0])
        sample = np.tile(sample, times)
    return sample[:length]


def mix_samples(sample_a: np.ndarray, sample_b: np.ndarray, trim_to: str = "a") -> np.ndarray:
    """
    @param sample_a: first audio-sample
    @param sample_b: second audio-sample
    @param trim_to: resulting length will be length of sample_a or sample_b,
                    if the sample not specified is shorter than the specified one,
                    it will be repeated to match the specified samples length
    @return: sum of both samples, normalized to avoid clipping
    """
    if trim_to == "b":
        sample_a, sample_b = sample_b, sample_a
    sample_b = match_length(sample_b, sample_a.shape[0])
    add = np.add(sample_a, sample_b)
    return normalize(add)


class AddNoise:
    def __init__(self, cfg, cwd):
        self.cfg = cfg
        self.cwd = cwd
        self.idx = 0

        urban_meta = pd.read_csv(cwd / self.cfg.urban_meta_path)
        self.noise_df = urban_meta[urban_meta["split"] == "test"].reset_index(drop=True)
        self.num_noises = len(self.noise_df.index)
        self.match_sr = create_match_to_speech(cfg.urban_sr, cfg.libri_sr)

    def load_noise(self) -> np.ndarray:
        current_idx = self.idx % self.num_noises
        self.idx += 1

        noise_path = self.noise_df.at[self.noise_df.index[current_idx], "PATH"]
        noise, sr = sf.read(f'{self.cwd}/{self.cfg.urban_path}/{noise_path}')
        return noise

    def __call__(self, src: str) -> tuple:
        noise = self.load_noise()
        speech, sr = sf.read(src)
        noise = self.match_sr(noise)
        noise = noise * self.cfg.artifact_scaling_factor
        mixed = mix_samples(speech, noise)
        return mixed, sr
