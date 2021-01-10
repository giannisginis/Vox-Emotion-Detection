import librosa
import librosa.display
import numpy as np
import os
import pickle


class AudioFeatures:

    def __init__(self, outpath):
        self.features = None
        self.outpath = outpath

    def extract_features(self, *feature_list, save_local=True, **kwargs):
        """
        Specify a list of features to extract, and a feature vector will be
        built for you for a given Audio sample.
        By default the extracted feature and class attributes will be saved in
        a local directory. This can be turned off with save_local=False.
        """
        extract_fn = dict(
            mfcc=self._extract_mfcc,
            spectral=self._extract_spectral_contrast,
            chroma=self._extract_chroma_stft,
            log_mel=self._extract_log_mel,
            kaldi=self._load_kaldi_feats
        )

        for feature in feature_list:
            extract_fn[feature](**kwargs)

        if save_local:
            self._save_local()

    def _update_features(self, features, concatenate=False):
        """
        Whenever a self._extract_xxx() method is called in this class,
        this function updates self.features attribute. There is a functionality to concatenate
        all the features if different types is chosen.
        """
        if concatenate:
            if self.features is not None:
                self.features = np.hstack([self.features, features])
            else:
                self.features = features
        else:
            self.features = features

    @staticmethod
    def load_audio(path_load, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5):
        return librosa.load(path_load, res_type=res_type, duration=duration, sr=sr, offset=offset)

    @staticmethod
    def trim_audio(audio):
        return librosa.effects.trim(audio)

    def _extract_log_mel(self, y=None, sr=22050, s=None, n_fft=2048, hop_length=512, win_length=None, window='hann',
                         center=True, pad_mode='reflect', power=2.0, n_mels=128, fmax=8000, ref=1.0, amin=1e-10,
                         top_db=80.0, pooling=True, concatenate=False):

        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, S=s, n_fft=n_fft, hop_length=hop_length,
                                                   win_length=win_length, window=window, center=center,
                                                   pad_mode=pad_mode, power=power, n_mels=n_mels,
                                                   fmax=fmax)
        mel_spect = librosa.power_to_db(mel_spect, ref=ref, amin=amin, top_db=top_db)
        if pooling:
            mel_spect = np.mean(mel_spect, axis=0)
        self._update_features(mel_spect, concatenate=concatenate)

    def _display_mel(self, y_axis='mel', fmax=20000, x_axis='time'):
        librosa.display.specshow(self.features, y_axis=y_axis, fmax=fmax, x_axis=x_axis)

    def _extract_mfcc(self, y=None, sr=22050, s=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, pooling=True,
                      concatenate=False):

        mfcc = librosa.feature.mfcc(y, sr=sr, S=s, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm, lifter=lifter)
        if pooling:
            mfcc = np.mean(mfcc, axis=0)
        self._update_features(mfcc, concatenate=concatenate)

    def _extract_spectral_contrast(self, y, sr, s=None, n_fft=2048, hop_length=512, win_length=None, window='hann',
                                   center=True, pad_mode='reflect', freq=None, fmin=200.0, n_bands=6, quantile=0.02,
                                   linear=False, concatenate=False):

        spec_con = librosa.feature.spectral_contrast(
            y=y, sr=sr, S=s, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center,
            pad_mode=pad_mode, freq=freq, fmin=fmin, n_bands=n_bands, quantile=quantile, linear=linear
        )
        self._update_features(spec_con, concatenate=concatenate)

    def _extract_chroma_stft(self, y, sr, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True,
                             dtype=None, pad_mode='reflect', norm=float('inf'), tuning=None, hop_length_chroma=512,
                             n_chroma=12, pooling=True, concatenate=False):

        stft = np.abs(
            librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center,
                         dtype=dtype, pad_mode=pad_mode))
        chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sr, norm=norm, n_fft=n_fft,
                                                  hop_length=hop_length_chroma, win_length=win_length,
                                                  window=window,
                                                  center=center, pad_mode=pad_mode, tuning=tuning,
                                                  n_chroma=n_chroma)

        if pooling:
            chroma_stft = np.mean(chroma_stft, axis=0)
        self._update_features(chroma_stft, concatenate=concatenate)

    def _save_local(self):
        """
        Saves features to disk as a pickle
        """
        filename = self.outpath + "/features.pkl"
        # if directory already exists leaves it unaltered and saves the file inside.
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self.features, f)

    def _load_kaldi_feats(self):  # TODO: revisit and implement kaldi_io
        pass
