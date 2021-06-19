import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import skimage.io
import os

class SignalToImageConverter:
    def __init__(self, hop_length, n_fft):
        """
        Creates a signal to image converter
        :param hop_length:
        :param n_fft:
        """
        self.hop_length = hop_length
        self.n_fft = n_fft

    def transform(self, signal, sample_rate):
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

    def to_image(self, specgram, sample_rate):
        librosa.display.specshow(specgram, sr=sample_rate, hop_length=self.hop_length)

    def save(self, specgram, root, filename):
        skimage.io.imsave(os.path.join(root, filename), specgram)


if __name__ == "__main__":
    converter = SignalToImageConverter(hop_length=512,
                                       n_fft=2048)

    signal, sample_rate = librosa.load('elo.wav', sr=22000)
    specgram = converter.transform(signal=signal,
                        sample_rate=sample_rate)
    print(specgram, type(specgram), specgram.shape)
