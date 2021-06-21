import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


class SignalToImageConverter:
    def __init__(self, hop_length, n_fft):
        """
        Creates a signal to image converter
        :param hop_length:
        :param n_fft:
        """
        self.hop_length = hop_length
        self.n_fft = n_fft

    def transform(self, signal):
        """
        Returns a spectrogram of a given signal
        :param signal:
        :param sample_rate:
        :return:
        """
        stft = librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length)
        specgram = np.abs(stft)
        abs_specgram = librosa.amplitude_to_db(specgram)
        return abs_specgram

    def _save_image(self, specgram, sample_rate, root, filename):
        image = librosa.display.specshow(specgram, sr=sample_rate, hop_length=self.hop_length)
        plt.savefig(os.path.join(root, filename), bbox_inches='tight', pad_inches=0.01)

    def tranform_save(self, signal, sample_rate, root, filename):
        specgram = self.transform(signal)
        self._save_image(specgram, sample_rate, root, filename)
        self._crop(root, filename)

    def _crop(self, root, filename):
        image = plt.imread(os.path.join(root, filename))
        plt.imsave(os.path.join(root, filename), image[2:image.shape[0]-5, 6:image.shape[1]-1])


if __name__ == "__main__":
    converter = SignalToImageConverter(hop_length=512,
                                       n_fft=2048)

    signal, sample_rate = librosa.load(r'/Signal-Processing-Project/audio_preprocessing/elo.wav', sr=22000)
    specgram = converter.transform(signal=signal)
    print(specgram, type(specgram), specgram.shape)
    converter.tranform_save(signal, sample_rate, r'C:\Users\gkrzy\projects\Signal-Processing-Project\audio_preprocessing', 'abcd.png')