import itertools
import pathlib
from random import uniform

import pandas as pd
import soundfile as sf
from librosa import resample
import numpy as np
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from scipy.signal import fftconvolve

from audio_utils import stereo_to_mono, normalize


def fetch_ir_paths(ir_path: str, cwd: str):
    path = pathlib.Path(cwd) / ir_path
    fetch_file_paths = lambda x: [(str(file.relative_to(path / x)),
                                   str(file.relative_to(path)))
                                  for file in x.glob("*.wav")]
    return itertools.chain.from_iterable([fetch_file_paths(directory) for directory in path.iterdir()])


def create_ir_dataframe(meta_path, ir_path: str, cwd: str):
    paths = fetch_ir_paths(ir_path, cwd)
    df_ir = pd.DataFrame.from_records(paths, columns=['FILE_NAME', 'PATH'])
    tmp, test = train_test_split(df_ir, test_size=0.2, random_state=42)
    train, dev = train_test_split(tmp, test_size=0.25, random_state=42)
    df_ir.loc[train.index, "SPLIT"] = "train"
    df_ir.loc[dev.index, "SPLIT"] = "dev"
    df_ir.loc[test.index, "SPLIT"] = "test"
    df_ir.to_csv(f'{cwd}/{meta_path}')
    return df_ir


class Reverberate:
    def __init__(self, cfg: DictConfig, cwd: str):
        self.cfg = cfg
        self.cwd = cwd
        self.idx = 0

        ir_df = pd.read_csv(f'{cwd}/{cfg.ir_meta_path}')
        self.ir_df = ir_df[ir_df["SPLIT"] == "test"]
        self.num_ir = len(self.ir_df)

    def convolve_impulse_response(self, speech: np.ndarray, impulse_response: np.ndarray, sr_speech: int, sr_impulse: int):
        impulse_response = resample(y=impulse_response.reshape(2, -1), orig_sr=sr_impulse, target_sr=sr_speech)
        impulse_response = impulse_response.reshape(-1, 2)

        convolved_signal = fftconvolve(speech, impulse_response, mode='full')
        convolved_signal = convolved_signal[0:speech.shape[0]] if self.cfg.ir_keep_duration else convolved_signal

        return convolved_signal

    def load_impulse(self):
        current_idx = self.idx % self.num_ir
        self.idx += 1

        ir_path = self.ir_df.at[self.ir_df.index[current_idx], "PATH"]
        impulse_response, sr = sf.read(f'{self.cwd}/{self.cfg.impulse_response_path}/{ir_path}')
        return impulse_response, sr

    def __call__(self, src) -> tuple:
        impulse_response, sr_imp = self.load_impulse()
        speech, sr_speech = sf.read(src, always_2d=True)
        wet_speech = self.convolve_impulse_response(speech, impulse_response, sr_speech, sr_imp)
        wet_speech = wet_speech * uniform(self.cfg.reverb_floor, self.cfg.reverb_ceil)
        mixed = np.add(speech, wet_speech)
        mixed = normalize(mixed)
        mono_mixed = stereo_to_mono(mixed)
        return mono_mixed, sr_speech
