import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import basename, join, splitext
from __Parameters import *
from __tag_tools import *


def calLabel(file, inaudio, total_dur, sr, ste, zcc, audio_map):
    # 10ms中有多少点
    seg_dur = int(DURATION * sr)
    if total_dur <= seg_dur:
        print("输入音频%s过短!" % file)
    else:
        seg_num = total_dur // seg_dur
        # print("共有:", seg_num, "个点")
        label = []
        for n in range(seg_num):
            start = n * seg_dur
            end = start + seg_dur
            if end > total_dur:
                print("Warning: Last 1 position not reached (less than =%fs)" % DURATION)

            label.append(tag_wav(ste, zcc, label, start, end, audio_map))

    return label



if __name__ == '__main__':

    for file in tqdm(os.listdir(FILEPATH)):
        allLabel=pd.DataFrame(columns=['song','time','label'])
        # 用vocals文件打标签
        file_name = file
        file = FILEPATH + file + '/mixture/vocals.wav'
        # file_name = file.split('/')[3]
        # 特征提取
        nf, frames, windown, nframes, framerate, sigs, times = prepare_data(file)
        # time = np.arange(0, nf) * (INC * 1.0 / framerate)
        # 计算zcc与ste
        zcc = get_zcc(nf, frames, windown)
        ste = get_ste(nf, frames, windown)
        # visualization(times, time, sigs, zcc, ste)
        # 计算zcc,ste和原signal的映射关系
        audio_map = map_audio_zcc_ste(zcc, ste, sigs)

        label = calLabel(file, sigs, nframes, framerate, ste, zcc, audio_map)

        for i in range(len(label)):
            allLabel.loc[i]=[file_name,"%.2f"%((i+1)/100),label[i]]
        allLabel.to_csv('./test_label/'+file_name+'_test_label.csv',index=False)

