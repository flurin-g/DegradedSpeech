import pathlib
from unittest import TestCase

from hydra.experimental import initialize, compose
import soundfile as sf

from dtln_de_noise import DtlnDeNoise

CWD = pathlib.Path(__file__).parent.parent.absolute()


class TestDtlnDeNoise(TestCase):
    @classmethod
    def setUpClass(cls):
        initialize("test_data/conf")
        cfg = compose("config")
        cls.cfg = cfg

    def setUp(self):
        self.freq_filter = DtlnDeNoise(cfg=self.cfg, cwd=CWD)

    def test_call(self):
        speech, sr = sf.read(f"{CWD}/tests/test_data/test-clean/61/70968/61-70968-0000.flac")
        path = f"{CWD}/tests/test_data/test-clean/61/70968/61-70968-0000.flac"
        de_noised, sr = self.freq_filter(path)
        #self.assertEqual(de_noised.shape, speech.shape)
        #print(de_noised.shape)
        sf.write("de_noised.flac", de_noised, sr)

