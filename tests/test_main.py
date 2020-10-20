from unittest import TestCase
import soundfile as sf
from hydra.experimental import initialize, compose

from add_noise import AddNoise
from dtln_de_noise import DtlnDeNoise
from freq_filter import FreqFilter
from main import create_copy_and_apply, get_task

import os

from reverberate import Reverberate


class Test(TestCase):
    @classmethod
    def setUpClass(cls):
        initialize("test_data/conf")
        cfg = compose("config")
        cls.cfg = cfg

    def test_get_task_add_noise(self):
        self.cfg.task = "add_noise"
        fx, source_directory, target_directory = get_task(self.cfg)
        self.assertEqual(fx, AddNoise)
        self.assertEqual(source_directory.split("/")[-1], "test-clean")
        self.assertEqual(target_directory.split("/")[-1], "noise")

    def test_get_task_add_reverberate(self):
        self.cfg.task = "reverberate"
        fx, source_directory, target_directory = get_task(self.cfg)
        self.assertEqual(fx, Reverberate)
        self.assertEqual(source_directory.split("/")[-1], "test-clean")
        self.assertEqual(target_directory.split("/")[-1], "reverb")

    def test_get_task_add_freq_filter(self):
        self.cfg.task = "freq_filter"
        fx, source_directory, target_directory = get_task(self.cfg)
        self.assertEqual(fx, FreqFilter)
        self.assertEqual(source_directory.split("/")[-1], "test-clean")
        self.assertEqual(target_directory.split("/")[-1], "freq_filter")

    def test_get_task_de_noise_noise(self):
        self.cfg.task = "de_noise"
        self.cfg.de_noise = "add_noise"
        fx, source_directory, target_directory = get_task(self.cfg)
        self.assertEqual(fx, DtlnDeNoise)
        self.assertEqual(source_directory.split("/")[-1], "noise")
        self.assertEqual(target_directory.split("/")[-1], "de_noised-add_noise")

    def test_get_task_de_noise_reverberate(self):
        self.cfg.task = "de_noise"
        self.cfg.de_noise = "reverberate"
        fx, source_directory, target_directory = get_task(self.cfg)
        self.assertEqual(fx, DtlnDeNoise)
        self.assertEqual(source_directory.split("/")[-1], "reverb")
        self.assertEqual(target_directory.split("/")[-1], "de_noised-reverberate")

    def test_get_task_de_noise_freq_filter(self):
        self.cfg.task = "de_noise"
        self.cfg.de_noise = "freq_filter"
        fx, source_directory, target_directory = get_task(self.cfg)
        self.assertEqual(fx, DtlnDeNoise)
        self.assertEqual(source_directory.split("/")[-1], "freq_filter")
        self.assertEqual(target_directory.split("/")[-1], "de_noised-freq_filter")

    def test_create_copy_and_apply(self):
        copy_fn = create_copy_and_apply(lambda x: (x, 16_000))
        print(type(copy_fn))
        assert(copy_fn, callable)

    def test_open_file(self):
        print(os.getcwd())
        file, sr = sf.read("test_data/test-clean/61/70968/61-70968-0000.flac")

    def test_copy_and_apply(self):
        def identity_audio(src):
            data, samplerate = sf.read(src)
            return data, samplerate
        cwd = os.getcwd()
        copy_fn = create_copy_and_apply(identity_audio)
        copy_fn(src=cwd + "/test_data/test-clean/61/70968/61-70968-0000.flac",
                dst=cwd + "/test_data/res.flac")
