import os

import sounddevice as sd
from scipy.io.wavfile import write


class AudioReader:
    def __init__(self, sample_rate=40000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels

    def record(self, seconds=1, filename=None, root=None):

        audio = sd.rec(int(seconds * self.sample_rate),
                       samplerate=self.sample_rate,
                       channels=self.channels)

        sd.wait()

        if filename:
            if root:
                write(os.path.join(root, filename), self.sample_rate, audio)
            else:
                write(os.path.join(os.path.dirname(__file__), filename), self.sample_rate, audio)

        return audio.reshape(-1)


if __name__ == "__main__":

    reader = AudioReader(sample_rate=22000,
                         channels=1)
    audio = reader.record(seconds=1)
    assert reader.sample_rate == 22000
    assert reader.channels == 1
    assert audio.shape == (22000,)
