# -*- coding: utf-8 -*-
# @Time    : 8/1/25
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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def mel_plot(mel_spect, sr):
    plt.ion()
    librosa.display.specshow(mel_spect, sr=sr, x_axis='time', y_axis='mel')
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time(s)')
    plt.title('Mel Spectrogram')
    plt.show()
    plt.pause(0.5)

def calculate_variance(data):
    n = len(data)
    if n < 2:
        return 0
    mean = sum(data) / n
    deviations = [(x - mean) for x in data]
    squared_deviations = [(x - mean)**2 for x in data]
    variance = sum(squared_deviations) / n
    return variance

def enframe(x, win, inc=None):
    nx = len(x)
    if isinstance(win, list) or isinstance(win, np.ndarray):
        nwin = len(win)
        nlen = nwin
    elif isinstance(win, int):
        nwin = 1
        nlen = win
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
    for i in range(nf):
        frameout[i, :] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list) or isinstance(win, np.ndarray):
        frameout = np.multiply(frameout, np.array(win))
    return frameout


def hanning_window(N):
    nn = [i for i in range(N)]
    return 0.5 * (1 - np.cos(np.multiply(nn, 2 * np.pi) / (N - 1)))

def right_shift(arr, shift_amount):
    shifted = np.roll(arr, shift_amount, axis=1)
    shifted[:, :shift_amount] = 0
    return shifted

def left_shift(arr, shift_amount):
    shifted = np.roll(arr, -shift_amount, axis=1)
    shifted[:, -shift_amount:] = 0
    return shifted

def STEn(x, win, inc):
    """
    :param x:
    :param win:
    :param inc:
    :return:
    """
    X = enframe(x, win, inc)
    s = np.multiply(X, X)
    return np.sum(s, axis=1)

def STEn_tea(x, win, inc):
    """
    :param x:
    :param win:
    :param inc:
    :return:
    """
    X = enframe(x, win, inc)
    X_last = right_shift(X, 1)
    X_next = left_shift(X, 1)
    s = np.multiply(X, X)-np.multiply(X_last, X_next)
    return np.sum(s, axis=1)

def int_sort(elem):
    return int(elem)

def wav_sort(elem):
    if elem.endswith(".wav"):
        wav_name = elem.split(".")[0]
    return int(wav_name)

def energy_get(path):
    data, fs = librosa.load(path, sr=None, mono=False)
    inc = 200
    wlen = 400
    win = hanning_window(wlen)
    N = len(data)
    time = [i / fs for i in range(N)]

    EN = STEn(data, win, inc)
    EN_tea = STEn_tea(data, win, inc)
    en_array = np.concatenate((EN.reshape(1,-1), EN_tea.reshape(1,-1)),axis=0)
    print('EN')
    return en_array

def extract_features(path):
    file = os.listdir(path)
    for f in file:
        name_wav = f.split('.')[0]
        data_mul_name = f'./energy/{name_wav}.txt'
        now_wav_path = os.path.join(path, f)
        print(now_wav_path)
        now_en = energy_get(now_wav_path)
        np.savetxt(data_mul_name, now_en)
    print('end')
    print('save')

if  __name__== "__main__":
    extract_features('../dataset')