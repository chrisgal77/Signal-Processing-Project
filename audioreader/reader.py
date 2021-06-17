import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os

class AudioReader():
    def __init__(self, sample_rate=40000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
    
    def record(self, seconds=1, filename=None, root=None, ):

        audio = sd.rec(int(seconds*self.sample_rate),
                        samplerate = self.sample_rate,
                        channels = self.channels)
        
        sd.wait()

        if filename:
            if root:
                write(os.path.join(root, filename), self.sample_rate, audio)
            else:
                write(os.path.join(os.path.dirname(__file__), filename), self.sample_rate, audio)
        
        return audio