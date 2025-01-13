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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def mel_plot(mel_spect, sr):
    plt.ion()
    #画mel谱图
    librosa.display.specshow(mel_spect, sr=sr, x_axis='time', y_axis='mel')
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time(s)')
    plt.title('Mel Spectrogram')
    plt.show()
    plt.pause(0.5)

def pause_time(path):
    data_time = {}
    sound = AudioSegment.from_wav(path)
    loudness = sound.dBFS
    data_time['duration_seconds'] = sound.duration_seconds
    chunks = silence.split_on_silence(sound,
                                      min_silence_len=400,
                                      silence_thresh=loudness * 1.3,
                                      keep_silence=400
                                      )
    data_time['num'] = len(chunks)-1
    sum_pause = 0
    for i in range(len(chunks)):
        print(len(chunks[i]))
        sum_pause = sum_pause + len(chunks[i])
    time_pause = len(sound) - sum_pause
    print('time_pause', time_pause)
    data_time['time_pause'] = time_pause
    data_time['time_pause/all'] = time_pause
    return time_pause

def calculate_variance(data):
    n = len(data)
    if n < 2:
        return 0
    mean = sum(data) / n
    deviations = [(x - mean) for x in data]
    squared_deviations = [(x - mean)**2 for x in data]
    variance = sum(squared_deviations) / n
    return variance

def pause_time_gnn(path):
    data_time = {}
    sound = AudioSegment.from_wav(path)
    loudness = sound.dBFS
    print('time', sound.duration_seconds)
    silence_list = silence.detect_silence(sound, silence_thresh=loudness * 1.3, min_silence_len=200)
    print('silent', silence_list)
    data_time['duration_seconds'] = sound.duration_seconds*1000
    data_time['num'] = len(silence_list)
    if data_time['num'] != 0:
        sum_pause = 0
        vary_list = []
        for i,silence_chunk in enumerate(silence_list):
            i_silence_time = silence_chunk[1] - silence_chunk[0]
            vary_list.append(i_silence_time)
            sum_pause = sum_pause + i_silence_time
        data_time['time_vary'] = calculate_variance(vary_list)
        data_time['time_pause'] = sum_pause
        if data_time['duration_seconds'] == 0:
            data_time['time_pause/all'] = 0
        else:
            data_time['time_pause/all'] = sum_pause/data_time['duration_seconds']
        if sum_pause == 0:
            data_time['time_pause/speak'] = 0
        else:
            if (data_time['duration_seconds']-sum_pause) == 0:
                data_time['time_pause/speak'] = 0
            else:
                data_time['time_pause/speak'] = sum_pause / (data_time['duration_seconds'] - sum_pause)
    else:
        data_time['time_vary'] = 0
        data_time['time_pause'] = 0
        data_time['time_pause/all'] = 0
        data_time['time_pause/speak'] = 0
    return data_time

def int_sort(elem):
    return int(elem)

def wav_sort(elem):
    if elem.endswith(".wav"):
        wav_name = elem.split(".")[0]
    return int(wav_name)

def extract_features(path):
    file = os.listdir(path)
    for f in file:
        path_now = os.path.join(path, f)
        wav_files = os.listdir(path_now)
        for wav_f in wav_files:

            data_pause_mul = np.zeros((6))
            name_wav = wav_f.split('.')[0]
            data_mul_name = f'./pause/{name_wav}.txt'
            now_wav_path = os.path.join(path_now, wav_f)
            print(now_wav_path)

            now_pause_time = pause_time_gnn(now_wav_path)
            data_pause_mul[0] = now_pause_time['duration_seconds']
            data_pause_mul[1] = now_pause_time['num']
            data_pause_mul[2] = now_pause_time['time_vary']
            data_pause_mul[3] = now_pause_time['time_pause']
            data_pause_mul[4] = now_pause_time['time_pause/all']
            data_pause_mul[5] = now_pause_time['time_pause/speak']
            np.savetxt(data_mul_name, data_pause_mul)
    print('end')
    print('save')



if  __name__== "__main__":
    extract_features('../dataset/speech_dir')
