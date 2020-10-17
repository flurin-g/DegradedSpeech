import pathlib
from unittest import TestCase

import numpy as np
import soundfile as sf
from hydra.experimental import initialize, compose


from add_noise import match_length, mix_samples, AddNoise, create_match_to_speech
from audio_utils import stereo_to_mono, normalize

CWD = pathlib.Path(__file__).parent.parent.absolute()


class TestFunctions(TestCase):
    def test_stereo_to_mono(self):
        sample, sr = sf.read("test_data/UrbanSound8K/audio/fold1/118279-8-0-5.wav")
        res = stereo_to_mono(sample)
        self.assertEqual(res.shape, (192000,))

    def test_create_match_to_speech(self):
        func = create_match_to_speech(noise_sr=44_100, speech_sr=16_000)
        assert(callable(func))

    def test_match_to_speech(self):
        sample, sr = sf.read("test_data/UrbanSound8K/audio/fold1/118279-8-0-5.wav")
        func = create_match_to_speech(noise_sr=44_100, speech_sr=16_000)
        res = func(sample)
        print(type(res))

    def test_match_length(self):
        sample = np.ones(128)
        length = 64
        res = match_length(sample, length)
        self.assertEqual(res.shape[0], 64)

    def test_normalize(self):
        sample = 2 * np.random.random_sample(128)
        res = normalize(sample)
        assert ((res >= -1).all() and (res <= 1).all())

    def test_mix_samples(self):
        sample_a = np.random.random_sample(128)
        sample_b = np.random.random_sample(256)
        res = mix_samples(sample_a, sample_b)
        assert ((res >= -1).all() and (res <= 1).all())
        self.assertEqual(res.shape, (128,))


class TestAddNoise(TestCase):
    @classmethod
    def setUpClass(cls):
        initialize("test_data/conf")
        cfg = compose("config")
        cls.cfg = cfg

    def setUp(self):
        self.add_noise = AddNoise(cfg=self.cfg, cwd=CWD)

    def test_load_noise(self):
        res, _ = self.add_noise.load_noise()
        self.assertEqual(res.shape, (192000, 2))
        self.assertEqual(self.add_noise.idx, 1)
        _, _ = self.add_noise.load_noise()
        self.assertEqual(self.add_noise.idx, 2)
        _, _ = self.add_noise.load_noise()
        self.assertEqual(self.add_noise.idx, 3)
        _, _ = self.add_noise.load_noise()
        self.assertEqual(self.add_noise.idx, 4)
        res2, _ = self.add_noise.load_noise()
        np.testing.assert_array_equal(res2, res)

    def test_call(self):
        sample, sr = sf.read(f"{CWD}/tests/test_data/test-clean/61/70968/61-70968-0000.flac")
        path = f"{CWD}/tests/test_data/test-clean/61/70968/61-70968-0000.flac"
        res, sr = self.add_noise(path)
        self.assertEqual(res.shape, sample.shape)
        sf.write("foo.flac", res, sr)



