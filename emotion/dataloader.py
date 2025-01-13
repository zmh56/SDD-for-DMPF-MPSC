# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

# ref in https://github.com/YuanGongND/ast
# @Time    : 13/1/25
# @Author  : Minghui Zhao
# @Affiliation  : Southeast university

import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, dataset_name, audio_conf, label_csv=None, audio_class=8):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')
        self.dataset_name = dataset_name
        self.dataset_json_file =dataset_json_file
        A = open(dataset_json_file)
        reanLines = []
        for line in A.readlines():
            line = line.strip()
            reanLines.append(line)
        A.close()
        self.audio_list = reanLines
        self.audio_class = audio_class

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        if self.dataset_name == 'ravdess':
            audio_name = self.audio_list[index]
            label_indices = np.zeros(self.audio_class)
            dataFile = './ravdess_ast_16k/' + audio_name

            fbank, mix_lambda = self._wav2fbank(dataFile)
            label_indices[int(audio_name.split('-')[2])-1] = 1.0

            label_indices = torch.FloatTensor(label_indices)
        elif self.dataset_name == 'lanzhou':
            audio_name = self.audio_list[index]
            label_indices = np.zeros(self.audio_class)
            dataFile = './lanzhou_16k_pydub/' + audio_name

            fbank, mix_lambda = self._wav2fbank(dataFile)
            if int(audio_name.split('_')[0]) >= 2020000:
                label_indices[0] = 1.0
            else:
                label_indices[1] = 1.0
            label_indices = torch.FloatTensor(label_indices)
        elif self.dataset_name == 'person100':
            audio_name = self.audio_list[index]
            label_indices = np.zeros(self.audio_class)
            # dataFile = '/home/she-56/PycharmProjects/speech_depression/A_f_s_ast_model_2023_06_01/att_sc_student/hospital_data_40person_16k/' + audio_name
            dataFile = '../dataset/speech_enhance/' + audio_name

            fbank, mix_lambda = self._wav2fbank(dataFile)

            if (audio_name.split('.')[0]).split('_')[-3] == 'N':
                label_indices[0] = 1.0
            else:
                label_indices[1] = 1.0

            label_indices = torch.FloatTensor(label_indices)
        elif self.dataset_name == 'daic':
            audio_name = self.audio_list[index]
            label_indices = np.zeros(self.audio_class)
            train_or_test = (self.dataset_json_file.split('/')[1]).split('_')[0]
            # if train_or_test == 'train':
            #     dataFile = './audio_data/train_data_daic/' + audio_name
            # elif train_or_test == 'test':
            #     dataFile = './audio_data/test_data_daic/' + audio_name
            dataFile = './' + audio_name
            fbank, mix_lambda = self._wav2fbank(dataFile)
            if int(audio_name.split('_')[2]) == 0:
                label_indices[0] = 1.0
            else:
                label_indices[1] = 1.0
            label_indices = torch.FloatTensor(label_indices)
        elif self.dataset_name == 'casia':
            audio_name = self.audio_list[index]
            label_indices = np.zeros(self.audio_class)
            dataFile = './CASIA_emo/' + audio_name

            fbank, mix_lambda = self._wav2fbank(dataFile)
            audio_name_endwise = audio_name.split('.')[0]
            tem_class = int(audio_name_endwise.split('_')[-1])
            if tem_class == 3 or tem_class == 4:
                tem_class = 1
            else:
                tem_class = 0
            label_indices[tem_class] = 1.0
        ######################################

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)


        # squeeze it back, it is just a trick to satisfy new torchaudio version
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        mix_ratio = min(mix_lambda, 1-mix_lambda) / max(mix_lambda, 1-mix_lambda)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label_indices, audio_name

    def __len__(self):
        return len(self.audio_list)
