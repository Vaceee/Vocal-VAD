import os
from os.path import basename, join, splitext
import librosa
from tqdm import tqdm


DURATION = 0.01
FILEPATH = './数据集/train/'
OUTBASEPATH = './dataset-temp/'



def split(file):
    file_path = splitext(file)[0].split('/',2)[2].rsplit('/',1)[0]
    file_name = splitext(basename(file))[0]
    inaudio, sr = librosa.load(file, sr=None, mono=False)  #sr=None To preserve the native sampling rate of the file
    if inaudio.shape == 2:    #双声道
        total_dur = len(inaudio[0])
    else:   # 单声道
        total_dur = len(inaudio)
    # print("total_dur:",total_dur)
    seg_dur = int(DURATION * sr)

    if total_dur <= seg_dur:
        print("输入音频%s过短!"%file)
    else:
        seg_num = total_dur // seg_dur
        for n in range(seg_num):
            start = n * seg_dur
            end = start + seg_dur
            if end > total_dur:
                print("Warning: Last 1 position not reached (less than =%fs)" % DURATION)

            out_dir = OUTBASEPATH + file_path + '/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            # librosa 只支持输出为 wav 格式
            librosa.output.write_wav(
                join(out_dir, file_name + "%05d.wav" % (n + 1)),
                inaudio[:,start:end],   # 多声道取第二维
                sr=sr,
                norm=False,
            )

for file in tqdm(os.listdir(FILEPATH)[:1]):
    file = FILEPATH + file + '/mixture.wav'
    # print(file)
    split(file)