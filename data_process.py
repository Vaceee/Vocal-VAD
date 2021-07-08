from os.path import basename, join, splitext
from __Parameters import *
from __tag_tools import *

import argparse
import os
import librosa
import numpy as np


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
            sig_split = inaudio[:, start:end]
            label = tag_wav(ste, zcc, label, start, end, audio_map)

            # out_dir = './dataset/' + file_path + '/'
            # if not os.path.exists(out_dir):
            #     os.makedirs(out_dir)
            # # librosa 只支持输出为 wav 格式

            # librosa.output.write_wav(
            #     join(out_dir, file_name + "%05d.wav" % (n + 1)),
            #     inaudio[:, start:end],   # 多声道取第二维
            #     sr=sr,
            #     norm=False,
            # )
    return label


nf, frames, windown, framerate, sigs, times = prepare_data(FILEPATH)
time = np.arange(0, nf) * (INC * 1.0 / framerate)
zcc = get_zcc(nf, frames, windown)
ste = get_ste(nf, frames, windown)
visualization(times, time, sigs, zcc, ste)
audio_map = map_audio_zcc_ste(zcc, ste, sigs)

print("sigs length", len(sigs))

label = []
file = FILEPATH
label = split(file, ste, zcc, label, audio_map)

num_1 = 0
num_0 = 0
for i in label:
    if i == 1:
        num_1 += 1
    else:
        num_0 += 1

print("label length:", len(label))
print("label 1 num:", num_1)
print("label 0 num:", num_0)

# for key, val in audio_map.items():
#     print(key, val)


# for file in os.listdir(FILEPATH):
#     file = FILEPATH + file + '/mixture.wav'
#     print(file)
#     split(file)
