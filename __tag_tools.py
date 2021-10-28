from abc import abstractproperty
from numpy.lib.function_base import append
# from spleeter.options import AudioOffsetOption
from __Parameters import *


import numpy as np
import wave as we
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf


def prepare_data(path):
    sample_rate, sigs = wf.read(path)
    f = we.open(path, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    # print("File params:", params)
    # calcuate time
    times = np.arange(nframes) / framerate
    str_data = f.readframes(nframes)

    wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))
    signal_length = len(wave_data)
    if signal_length <= WLEN:
        nf = 1
    else:
        nf = int(np.ceil((1.0 * signal_length - WLEN + INC) / INC))
    # 所有帧加起来总的铺平后的长度
    pad_length = int((nf - 1) * INC + WLEN)
    # 不够的长度用0填补
    zeros = np.zeros((pad_length - signal_length,))
    pad_signal = np.concatenate((wave_data, zeros))

    indices = np.tile(np.arange(0, WLEN), (nf, 1)) + \
        np.tile(np.arange(0, nf * INC, INC), (WLEN, 1)).T

    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]
    windown = np.hamming(WLEN)
    return nf, frames, windown, nframes, framerate, sigs, times


def get_zcc(nf, frames, windown):
    zcc = np.zeros(nf)
    for i in range(nf):
        a = frames[i: i + 1]
        b = windown * a[0]
        for j in range(WLEN - 1):
            if b[j] * b[j + 1] < 0 and abs(b[j] - b[j + 1]) > ZCC_DELTA:
                zcc[i] = zcc[i] + 1
    return zcc


def get_ste(nf, frames, windown):
    ste = np.zeros(nf)
    e = np.zeros(nf)
    for i in range(0, nf):
        a = frames[i: i + 1]
        b = a[0] * windown
        e[i] = np.sum(abs(b))
        c1 = np.square(b)
        ste[i] = np.sum(c1)
    e = e * 1.0 / (max(abs(e)))
    ste = ste * 1.0 / (max(abs(ste)))
    return ste


def tag_wav(ste, zcc, label, start, end, audio_map):
    num = 0
    for i in range(start, end):
        idx = audio_map[i]
        if ste[idx] > THRESHOLD_STE and zcc[idx] < THRESHOLD_ZCC:
            num += 1
    prob = num / (end - start)
    return prob


def visualization(times, time, sigs, zcc, ste):
    # 信号波形
    plt.figure(figsize=(15, 8))
    plt.subplot(4, 1, 1)
    plt.title('Time Domain')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    plt.plot(times, sigs, c='blue', label='Signal')
    plt.legend(['audio'])

    # 短时过零率
    plt.subplot(4, 1, 2)
    plt.xlabel('Time')
    plt.ylabel('zero-crossing number')
    plt.plot(time, zcc, c='red')
    plt.grid(linestyle=':')
    plt.legend(['Zero Crossing Rate'])

    # 短时能量
    plt.subplot(4, 1, 3)
    plt.xlabel('Time')
    plt.ylabel('short-time energy')
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    plt.plot(time, ste, c='orange')
    plt.legend(['Energy'])

    # 短时幅度
    # plt.subplot(4, 1, 4)
    # plt.xlabel('Time')
    # plt.ylabel('short-time magnitude')
    # plt.tick_params(labelsize=10)
    # plt.grid(linestyle=':')
    # plt.plot(time, e, c='blue')
    # plt.legend(['Magnitude'])

    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    plt.rcParams['figure.dpi'] = 500  # 分辨率
    plt.savefig('./img/ZCC_STE.png')
    plt.show()


def map_audio_zcc_ste(zcc, ste, sigs):
    audio_map = {}
    value_length = len(zcc)
    ste_length = len(ste)
    # print("zcc length:", len(zcc))
    # print("ste length:", len(ste))
    key_length = len(sigs)
    interval = key_length // value_length
    # print("interval:", interval)
    __up = interval
    __cur_idx = 0
    for i in range(len(sigs)):
        if i > __up:
            __up += interval
            if __cur_idx < value_length - 1:
                __cur_idx += 1
        audio_map[i] = __cur_idx
    return audio_map

# print("num_ste_1:", num_ste_1)
# print("STE length:", len(ste))
# print("ZCC length:", len(zcc))
# print("signal length:", len(time))
# print("label length:", len(label))
# print("1 num:", num_1)
# print("0 num:", num_0)
