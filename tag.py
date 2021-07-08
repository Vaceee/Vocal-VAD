import numpy as np
import wave as we
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf

path = 'vocals.wav'
wlen = 480
inc = 120
sample_rate, sigs = wf.read(path)
times = np.arange(len(sigs)) / sample_rate
f = we.open(r'vocals.wav', "rb")
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
print(params)

str_data = f.readframes(nframes)
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data = wave_data * 1.0 / (max(abs(wave_data)))
# print(wave_data[:10])
signal_length = len(wave_data)
if signal_length <= wlen:
    nf = 1
else:
    nf = int(np.ceil((1.0 * signal_length - wlen + inc) / inc))
# print(nf)
pad_length = int((nf - 1) * inc + wlen)
zeros = np.zeros((pad_length - signal_length,))
pad_signal = np.concatenate((wave_data, zeros))
indices = np.tile(np.arange(0, wlen), (nf, 1)) + \
    np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T
# print(indices[:2])
indices = np.array(indices, dtype=np.int32)
frames = pad_signal[indices]
windown = np.hamming(wlen)
d = np.zeros(nf)
x = np.zeros(nf)
c = np.zeros(nf)
e = np.zeros(nf)
for i in range(nf):
    a = frames[i: i + 1]
    b = windown * a[0]
    for j in range(wlen - 1):
        if b[j] * b[j + 1] < 0:
            c[i] = c[i] + 1

time = np.arange(0, nf) * (inc * 1.0 / framerate)
for i in range(0, nf):
    a = frames[i: i + 1]
    b = a[0] * windown
    e[i] = np.sum(abs(b))
    c1 = np.square(b)
    d[i] = np.sum(c1)
e = e * 1.0 / (max(abs(e)))
d = d * 1.0 / (max(abs(d)))
# print(d)

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
plt.plot(time, c, c='blue')
plt.grid(linestyle=':')
plt.legend(['Zero Crossing Rate'])

# 短时能量
plt.subplot(4, 1, 3)
plt.xlabel('Time')
plt.ylabel('short-time energy')
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')
plt.plot(time, d, c='blue')
plt.legend(['Energy'])

# 短时幅度
plt.subplot(4, 1, 4)
plt.xlabel('Time')
plt.ylabel('short-time magnitude')
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')
plt.plot(time, e, c='blue')
plt.legend(['Magnitude'])

plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.savefig('image.png')
plt.show()
