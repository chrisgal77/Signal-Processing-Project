from utils import SignalToImageConverter

from tqdm import tqdm
import librosa
import os
#
#
# def rename_files():
#     for i, file_name in enumerate(os.listdir(r"D:\python-projects\data\other")):
#         destination="other" + str(i) + ".wav"
#         source=r'D:\python-projects\data\other\\'+ file_name
#         destination=r'D:\python-projects\data\other\\'+ destination
#         os.rename(source, destination)
#
#
# rename_files()


def transform():
    converter = SignalToImageConverter(hop_length=512,
                                       n_fft=2048)
    for name in ['other']:
        for i, file_name in tqdm(enumerate(os.listdir(fr"C:\Users\Gal\Desktop\data\{name}\\"))):
            if name == 'other' and i < 876:
                continue
            source = fr'C:\Users\Gal\Desktop\data\{name}\\'
            signal, sr = librosa.load(source+file_name, sr=22000)
            converter.tranform_save(signal, sr, fr'C:\Users\Gal\Desktop\transformed\{name}\\', f'{name}'+str(i)+'.png')


transform()
