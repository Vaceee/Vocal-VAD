from os.path import basename, join, splitext
from tqdm import tqdm

import joblib
from __Parameters import *
from __tag_tools import *

import argparse
import os
import librosa
import numpy as np
import pandas as pd


def split(file, ste, zcc, label, audio_map):
    # file_path = splitext(file)[0].split('/', 2)[2].rsplit('/', 1)[0]
    # file_name = splitext(basename(file))[0]
    # sr=None To preserve the native sampling rate of the file

    inaudio, sr = librosa.load(file, sr=None, mono=False)
    print("采样率为：", sr)
    if inaudio.shape == 1:
        total_dur = len(inaudio)
    else:   # 多声道
        total_dur = len(inaudio[0])
    print("total_dur:", total_dur)
    # 10ms中有多少点
    seg_dur = int(DURATION * sr)
    if total_dur <= seg_dur:
        print("输入音频%s过短!" % file)
    else:
        seg_num = total_dur // seg_dur
        print("共有:", seg_num, "个点")
        for n in range(seg_num):
            start = n * seg_dur
            end = start + seg_dur
            if end > total_dur:
                print("Warning: Last 1 position not reached (less than =%fs)" % DURATION)
            label = tag_wav(ste, zcc, label, start, end, audio_map)
            # out_dir = OUTPATH + file_path + '/'
            # if not os.path.exists(out_dir):
            #     os.makedirs(out_dir)
    return label


file_name2idx = joblib.load('./para/' + 'file_name2idx.pkl')
file_name2vocals = joblib.load('./para/' + 'file_name2vocals.pkl')

all_label = pd.DataFrame(columns=['song', 'time', 'vocals_probability'])
start = 0

label = []
i = 0
print(FILEPATH)
for file in tqdm(os.listdir(FILEPATH)):
    # 用vocals文件打标签
    file = FILEPATH + file + '/vocals.wav'
    file_name = file.split('/')[3]
    print(file)
    # 特征提取
    nf, frames, windown, framerate, sigs, times = prepare_data(file)
    time = np.arange(0, nf) * (INC * 1.0 / framerate)
    # 计算zcc与ste
    zcc = get_zcc(nf, frames, windown)
    ste = get_ste(nf, frames, windown)
    # visualization(times, time, sigs, zcc, ste)
    audio_map = map_audio_zcc_ste(zcc, ste, sigs)
    print("sigs length:", len(sigs))

    label = split(file, ste, zcc, audio_map)

    length = len(label)
    for i in range(length):
        all_label.loc[start + i] = [file_name, "%05d" % (i + 1), int(label[i])]
    start += length

all_label.to_csv('./test_label.csv', index=False)
