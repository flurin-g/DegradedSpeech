from unittest import TestCase
import soundfile as sf
from main import create_copy_and_apply

import os


class Test(TestCase):
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
