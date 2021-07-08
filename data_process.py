from os.path import basename, join, splitext
from __Parameters import *
from __tag_tools import *

import argparse
import os
import librosa
import numpy as np


def split(file, ste, zcc, label, audio_map):
    file_path = splitext(file)[0].split('/', 2)[2].rsplit('/', 1)[0]
    file_name = splitext(basename(file))[0]
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
            out_dir = OUTPATH + file_path + '/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

    return label


label = []
for file in os.listdir(FILEPATH):
    file = FILEPATH + file + '/mixture.wav'
    nf, frames, windown, framerate, sigs, times = prepare_data(file)
    time = np.arange(0, nf) * (INC * 1.0 / framerate)
    zcc = get_zcc(nf, frames, windown)
    ste = get_ste(nf, frames, windown)
    visualization(times, time, sigs, zcc, ste)
    audio_map = map_audio_zcc_ste(zcc, ste, sigs)

    
    print(file)
    label = split(file, ste, zcc, label, audio_map)

