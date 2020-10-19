import numpy as np
import soundfile as sf
import tensorflow as tf

class DtlnDeNoise:
    def __init__(self, cfg, cwd):
        self.cfg = cfg
        self.cwd = cwd

        # fixed value, need to retrain if changed
        self.block_len = 512
        self.block_shift = 128

        # load model
        self.model = tf.saved_model.load(f'{cwd}/dtln_saved_model')
        self.infer = self.model.signatures["serving_default"]

    def de_noise(self, speech) -> np.ndarray:
        # preallocate output audio
        out_file = np.zeros((len(speech)))
        # create buffer
        in_buffer = np.zeros((self.block_len))
        out_buffer = np.zeros((self.block_len))
        # calculate number of blocks
        num_blocks = (speech.shape[0] - (self.block_len - self.block_shift)) // self.block_shift
        # iterate over the number of blcoks
        for idx in range(num_blocks):
            # shift values and write to buffer
            in_buffer[:-self.block_shift] = in_buffer[self.block_shift:]
            in_buffer[-self.block_shift:] = speech[idx * self.block_shift:(idx * self.block_shift) + self.block_shift]
            # create a batch dimension of one
            in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
            # process one block
            out_block = self.infer(tf.constant(in_block))['conv1d_1']
            # shift values and write to buffer
            out_buffer[:-self.block_shift] = out_buffer[self.block_shift:]
            out_buffer[-self.block_shift:] = np.zeros((self.block_shift))
            out_buffer += np.squeeze(out_block)
            # write block to output file
            out_file[idx * self.block_shift:(idx * self.block_shift) + self.block_shift] = out_buffer[:self.block_shift]

        return out_file

    def __call__(self, src: str) -> tuple:
        speech, sr = sf.read(src)
        de_noised = self.de_noise(speech)
        return de_noised, sr