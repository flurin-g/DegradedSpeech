import pathlib
from unittest import TestCase

import soundfile as sf
from hydra.experimental import initialize, compose
from librosa import resample
from scipy.signal import fftconvolve
from sklearn.model_selection import train_test_split

from audio_utils import stereo_to_mono
from reverberate import Reverberate, fetch_ir_paths, create_ir_dataframe

CWD = pathlib.Path(__file__).parent.parent.absolute()


class TestFunctions(TestCase):
    def test_fetch_ir_paths(self):
        res = fetch_ir_paths(ir_path="tests/test_data/ImpulseResponses", cwd=CWD)
        print(list(res))

    def test_create_ir_dataframe(self):
        df_ir = create_ir_dataframe(meta_path="tests/test_data/ir_meta.csv",
                                    ir_path="tests/test_data/ImpulseResponses",
                                    cwd=CWD)
        print(f'train:\n{df_ir[df_ir["SPLIT"] == "train"]}')
        print(f'dev:\n{df_ir[df_ir["SPLIT"] == "dev"]}')
        print(f'test:\n{df_ir[df_ir["SPLIT"] == "test"]}')


class TestReverberate(TestCase):
    @classmethod
    def setUpClass(cls):
        initialize("test_data/conf")
        cfg = compose("config")
        cls.cfg = cfg

    def setUp(self):
        self.reverberate = Reverberate(cfg=self.cfg, cwd=CWD)

    def test_impulse_response(self):
        speech, sr_speech = sf.read(f"test_data/test-clean/61/70968/61-70968-0000.flac", always_2d=True)
        impulse_response, sr_resp = sf.read("test_data/ImpulseResponses/1st-baptist-nashville/stereo/1st_baptist_nashville_balcony.wav")

        impulse_response = resample(y=impulse_response.reshape(2, -1), orig_sr=sr_resp, target_sr=sr_speech)
        impulse_response = impulse_response.reshape(-1,2)

        convolved_signal = fftconvolve(speech, impulse_response, mode='full')

        mono = stereo_to_mono(convolved_signal)

        sf.write("conv_test.flac", mono, sr_speech)

    def test_fetch_impulse_response_filenames(self):
        print(self.reverberate.num_ir)

    def test_load_impulse(self):
        res, sr = self.reverberate.load_impulse()
        self.assertEqual(sr, 44100)

    def test_call(self):
        path = f"{CWD}/tests/test_data/test-clean/61/70968/61-70968-0000.flac"
        wet, sr = self.reverberate(path)
        sf.write("wet.flac", wet, sr)


