# -*- coding: utf-8 -*-
"""Data Loader"""

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from Featurizer.featurizer import AudioFeatures


class Dataloader(AudioFeatures):
    """Data Loader class"""

    def __init__(self, metadata, path_folder, outpath):
        super().__init__(outpath)
        self.emotion = []
        self.gender = []
        self.actor = []
        self.file_path = []
        self.combined_data = None

        self.path_folder = path_folder
        self.folders_main = os.listdir(self.path_folder)
        self.outpath = outpath
        self.metadata = metadata
        self.counter = 0
        self.train = None

        self._create_dir(self.outpath)

    @staticmethod
    def _create_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def load_data(self, save2disk=False):
        """
        Loads dataset from path
        :return: None
        """
        for folder in tqdm(self.folders_main, desc="Read files from folders"):
            path_in = self.path_folder + '/{0}'.format(folder)
            files = os.listdir(path_in)  # iterate over Actor folders

            for file in files:  # go through files in Actor folder
                part = file.split('.')[0].split('-')
                self.emotion.append(int(part[2]))
                self.actor.append(int(part[6]))
                bg = int(part[6])
                if bg % 2 == 0:
                    bg = "female"
                else:
                    bg = "male"
                self.gender.append(bg)
                self.file_path.append("/".join((self.path_folder, folder, file)))

        self._data2pandas(save2disk)

    def _data2pandas(self, save2disk=False):
        """
        PUT EXTRACTED LABELS WITH FILEPATH INTO DATAFRAME
        :return: None
        """

        audio_df = pd.DataFrame(self.emotion)
        audio_df = audio_df.replace(self.metadata)
        audio_df = pd.concat([pd.DataFrame(self.gender), audio_df, pd.DataFrame(self.actor)], axis=1)
        audio_df.columns = ['gender', 'emotion', 'actor']
        self.combined_data = pd.concat([audio_df, pd.DataFrame(self.file_path, columns=['path'])], axis=1)

        if save2disk:
            self.combined_data.to_csv(self.outpath + '/combined_data.csv')

    def feature_extraction(self, feature_type='raw_audio', pooling=True):
        self._data2pandas()
        # ITERATE OVER ALL AUDIO FILES AND EXTRACT LOG MEL SPECTROGRAM MEAN VALUES INTO DF FOR MODELING
        df = pd.DataFrame(columns=['mel_spectrogram'])

        counter = 0

        for index, path in enumerate(tqdm(self.combined_data.path, desc="feature extraction from files")):
            X, sample_rate = self.load_audio(path, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5)

            # get the mel-scaled spectrogram (Transform both the y-axis (frequency) to log scale, and the “color” axis
            # (amplitude) to Decibels, which is kinda the log scale of amplitudes.)

            if feature_type != "raw_audio":
                self.extract_features(feature_type, save_local=False, y=X, sr=sample_rate, pooling=pooling)
            elif feature_type == "raw_audio":
                self.features = X

            df.loc[counter] = [self.features]
            counter = counter + 1

        # TURN ARRAY INTO LIST AND JOIN WITH COMBINE DF TO GET CORRESPONDING EMOTION LABELS
        self.combined_data = pd.concat([self.combined_data, pd.DataFrame(df['mel_spectrogram'].values.tolist())],
                                       axis=1)
        self.combined_data = self.combined_data.fillna(0)

    def preprocess_data(self, normalize=True, test_size=0.2, split=True, encoder='OneHotEncoder'):
        """
        Preprocess and splits into training and test
        :param normalize: flag for normalization
        :param test_size: test size for split
        :param split: triggers the split of the data based on test size
        :param encoder: define the encoder type of the labels. 'LabelEncoder' || 'OneHotEncoder'
        :return: X_train, X_test, y_train, y_test
        """

        if split:
            self._split_data(test_size)
        else:
            self.train = self.combined_data

        X_train = self.train.iloc[:, 4:]
        y_train = pd.DataFrame(self.train.emotion)

        X_test = self.test.iloc[:, 4:]
        y_test = pd.DataFrame(self.test.emotion)

        # label encoder
        y_train, y_test, label2index, index2label = self._label_encoder(y_train, y_test,
                                                                        method=encoder)

        if normalize:
            X_train, X_test = self._normalize(X_train, X_test)

        X_train, y_train = self._pandas2numpy(X_train, y_train)
        X_test, y_test = self._pandas2numpy(X_test, y_test)

        return X_train, X_test, y_train, y_test

    def _split_data(self, test_size=0.2):
        # TRAIN TEST SPLIT DATA
        self.train, self.test = train_test_split(self.combined_data, test_size=test_size, random_state=0,
                                                 stratify=self.combined_data[['emotion', 'gender', 'actor']])

    @staticmethod
    def _normalize(x_train, x_test):

        """
        NORMALIZE DATA
        :param x_train: train features
        :param x_test: test features
        :return:  X_train, X_test
        """

        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

        return x_train, x_test

    @staticmethod
    def _pandas2numpy(x, y):
        """
        TURN DATA INTO ARRAYS FOR THE MODEL
        :param x: features
        :param y: labels
        :return: X, y
        """

        x = np.array(x)
        y = np.array(y)

        return x, y

    @staticmethod
    def _label_encoder(y_train, y_test, method='LabelEncoder'):
        """
        Labels to one hot vector
        :param y_train: train labels
        :param y_test: test labels
        :param method: 'LabelEncoder' || 'OneHotEncoder'
        :return: y_train, y_test, label2index, index2label
        """
        label2index, index2label = {}, {}

        if method == 'LabelEncoder':
            # creating instance of LabelEncoder
            enc = LabelEncoder()
            y_train = enc.fit_transform(y_train)
            y_test = enc.fit_transform(y_test)
            for i in range(enc.classes_):
                print(i)
            label2index = {d: i for i, d in enumerate(enc.classes_)}
            index2label = {i: d for i, d in enumerate(enc.classes_)}
        elif method == 'OneHotEncoder':
            # creating instance of one-hot-encoder
            enc = OneHotEncoder(handle_unknown='ignore')
            y_train = pd.DataFrame(enc.fit_transform(y_train).toarray())
            y_test = pd.DataFrame(enc.fit_transform(y_test).toarray())
            label2index = {d: i for i, d in enumerate(enc.categories_[0])}
            index2label = {i: d for i, d in enumerate(enc.categories_[0])}

        return y_train, y_test, label2index, index2label
