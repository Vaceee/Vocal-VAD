import os
from tqdm import tqdm

FILEPATH = './数据集/test/'

for file in tqdm(os.listdir(FILEPATH)):
    file_path = FILEPATH + file + '/mixture.wav'
    os.system('spleeter separate -p spleeter:2stems -o "dataset/spleeter/' + file + '" "'  + file_path + '"')