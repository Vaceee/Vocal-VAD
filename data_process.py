import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import basename, join, splitext
from __Parameters import *
from __tag_tools import *


DURATION = 0.01
FILEPATH = './数据集/train/'


def split(file, ste, zcc, audio_map):
    # file_path = splitext(file)[0].split('/', 2)[2].rsplit('/', 1)[0]
    # file_name = splitext(basename(file))[0]
    # set sr=None To preserve the native sampling rate of the file

    inaudio, sr = librosa.load(file, sr=None, mono=False)
    # print("采样率为：", sr)
    if inaudio.shape == 1:
        total_dur = len(inaudio)
    else:   # 多声道
        total_dur = len(inaudio[0])
    # print("total_dur:", total_dur)
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



if __name__ == '__main__':

    allLabel=pd.DataFrame(columns=['song','time','label'])
    start=0

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
        # start:start+length行文件名全部赋值
        # allLabel['song'].iloc[start:start+length] = file_name
        for i in range(start,start+length):
            allLabel.loc[i]=[file_name,"%05d"%(i+1),int(label[i])]
            # allLabel['time'].iloc[start+i] = "%05d.wav"%(i+1)
            # allLabel['label'].iloc[start+i] = label[i]
        start += length
    allLabel.to_csv('./train_label.csv',index=False)



