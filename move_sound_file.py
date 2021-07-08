import os
import joblib

from __Parameters import *

i = 0
file_name2idx = {}
sound_name = []

for file in os.listdir(FILEPATH):
    file_name2idx[file] = 'mixture_' + str(i) + '.wav'
    sound_name.append('mixture_' + str(i) + '.wav')
    file_path = FILEPATH + file + '/mixture.wav'
    print('mixture_' + str(i) + '.wav')
    i += 1

joblib.dump(file_name2idx, './para/' + 'file_name2idx.pkl')
joblib.dump(sound_name, './para/' + 'sound_name.pkl')

