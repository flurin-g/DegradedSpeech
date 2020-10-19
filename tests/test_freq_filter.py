import pathlib
from unittest import TestCase

from hydra.experimental import initialize, compose
import soundfile as sf
from freq_filter import FreqFilter, butter_bandpass_filter, butter_bandpass

CWD = pathlib.Path(__file__).parent.parent.absolute()


class TestFreqFilter(TestCase):
    @classmethod
    def setUpClass(cls):
        initialize("test_data/conf")
        cfg = compose("config")
        cls.cfg = cfg

    def setUp(self):
        self.freq_filter = FreqFilter(cfg=self.cfg, cwd=CWD)

    def test_run(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.signal import freqz

        # Sample rate and desired cutoff frequencies (in Hz).
        fs = 5000.0
        lowcut = 500.0
        highcut = 1250.0

        # Plot the frequency response for a few different orders.
        plt.figure(1)
        plt.clf()
        for order in [3, 6, 9]:
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

        plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
                 '--', label='sqrt(0.5)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')

        # Filter a noisy signal.
        T = 0.05
        nsamples = int(T * fs)
        t = np.linspace(0, T, nsamples, endpoint=False)
        a = 0.02
        f0 = 600.0
        x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
        x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
        x += a * np.cos(2 * np.pi * f0 * t + .11)
        x += 0.03 * np.cos(2 * np.pi * 2000 * t)
        plt.figure(2)
        plt.clf()
        plt.plot(t, x, label='Noisy signal')

        y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
        plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
        plt.xlabel('time (seconds)')
        plt.hlines([-a, a], 0, T, linestyles='--')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc='upper left')

        plt.show()

    def test_calc_params(self):
        low_bound = self.cfg.hp_low
        up_bound = low_bound + self.cfg.bandwidth
        for i in range(20):
            low, high, order = self.freq_filter.calc_params()
            self.assertTrue(low_bound <= low <= up_bound)
            self.assertTrue(order, self.cfg.order)
            print(low, high, order)

    def test_call(self):
        speech, sr = sf.read(f"{CWD}/tests/test_data/test-clean/61/70968/61-70968-0000.flac")
        path = f"{CWD}/tests/test_data/test-clean/61/70968/61-70968-0000.flac"
        filtered, sr = self.freq_filter(path)
        self.assertEqual(filtered.shape, speech.shape)
        print(filtered.shape)
        sf.write("filter.flac", filtered, sr)

