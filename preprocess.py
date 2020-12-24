# Author Ioannis Gkinis, 2020

# imports
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import re
import os

dicts = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful',
         '07': 'disgust', '08': 'surprised'}


