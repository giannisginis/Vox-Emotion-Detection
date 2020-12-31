# Author Ioannis Gkinis, 2020
# i used this cmd to sync two directories of dataset ---> rsync -avhu --progress source destination
# imports
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from tqdm import tqdm


class MelFeatures:

    def __init__(self):
        self.mel_spect = None

    def compute_mel(self, y, sr, n_fft=1024, hop_length=100):
        self.mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        self.mel_spect = librosa.power_to_db(self.mel_spect, ref=np.max)

    def display_mel(self, y_axis='mel', fmax=20000, x_axis='time'):
        librosa.display.specshow(self.mel_spect, y_axis=y_axis, fmax=fmax, x_axis=x_axis)

    @staticmethod
    def load_audio(path_load):
        return librosa.load(path_load)

    @staticmethod
    def trim_audio(audio):
        return librosa.effects.trim(audio)


class AudioInfo(MelFeatures):

    def __init__(self, n_fft=1024, hop_length=100, y_axis='mel', fmax=20000, x_axis='time'):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.y_axis = y_axis
        self.fmax = fmax
        self.x_axis = x_axis

        self.y = None
        self.sr = None
        self.yt = None
        self.srt = None

    def _load_audio(self, audio_path):
        self.y, self.sr = self.load_audio(audio_path)

    def _trim_audio(self):
        self.yt, self.srt = self.trim_audio(self.y)

    def calculate_mel(self):
        self.compute_mel(self.y, self.sr, self.n_fft, self.hop_length)

    def show_mel(self):
        self.display_mel(self.y_axis, self.fmax, self.x_axis)


class PreprocessData(AudioInfo):

    def __init__(self, metadata, path_folder, outpath):
        super().__init__()
        self.path_folder = path_folder
        self.folders_main = os.listdir(self.path_folder)
        self.outpath = outpath
        self.metadata = metadata
        self.counter = 0

        self._create_dir(self.outpath)

    @staticmethod
    def _create_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def process_audios(self):  # TODO add tqdm
        for folders in self.folders_main:
            path_in = self.path_folder + '/{0}'.format(folders)
            files_sub = os.listdir(path_in)
            pbar = tqdm(files_sub, desc="second loop")
            for file in pbar:
                numbers = re.findall('\d+', file)
                emotion = self.metadata[numbers[2]]
                pbar.set_description(numbers[2] + " " + emotion)

                path_save = self.outpath + '/{0}'.format(emotion)
                self._create_dir(path_save)
                outfile = path_save + '/{1}.jpeg'.format(emotion, file.split(".")[0])

                path_load = '{0}/{1}'.format(path_in, file)

                # audio load and computation of mel
                self._load_audio(path_load)
                self._trim_audio()
                self.calculate_mel()
                self.display_mel()

                # save mel to folder
                plt.savefig(outfile)
