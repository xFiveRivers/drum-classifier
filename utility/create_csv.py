import csv
import os
import re

data_folder = 'data/wavefiles'
wav_files = os.listdir(data_folder)
concat_result = ' '.join(wav_files)
class_list = re.findall(r'(\S+)_\d+\.wav', concat_result)

with open('data/samples.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["sample", "class"])
    writer.writerows(zip(wav_files, class_list))