# @Time    : 13/1/25
# @Author  : Minghui Zhao
# @Affiliation  : Southeast university

import wave
import numpy as np
import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

import soundfile as sf

from pydub import AudioSegment
# from pydub.silence import split_on_silence
from pydub import silence

def calculate_jitter_shimmer(audio, fs):
    # Shimmer
    audio_frames = librosa.util.frame(audio, frame_length=int(0.02 * fs), hop_length=int(0.01 * fs)).T
    amplitude = np.max(audio_frames, axis=1)
    shimmer = np.abs(np.diff(amplitude))
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=75, fmax=600,
                                                 frame_length=int(0.02 * fs), hop_length=int(0.01 * fs), center=False)
    f0 = f0[voiced_flag > 0]
    amplitude = amplitude[voiced_flag > 0]
    j_s_array = np.concatenate((f0.reshape(1, -1), amplitude.reshape(1, -1)), axis=0)
    return j_s_array

def extract_features(path):
    file = os.listdir(path)
    for f in file:
        name_wav = f.split('.')[0]
        data_mul_name = f'./tremor/{name_wav}.txt'
        now_wav_path = os.path.join(path, f)
        print(now_wav_path)
        audio, fs = librosa.load(now_wav_path, sr=None)
        j_s_array = calculate_jitter_shimmer(audio, fs)
        np.savetxt(data_mul_name, j_s_array)
    print('end')
    print('save')

if  __name__== "__main__":
    extract_features('../dataset')