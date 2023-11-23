import os
import sys

inst = str(sys.argv[1])

file_delete_dir = f'data/{inst}/{inst}_delete/'
file_delete_path = f'data/{inst}/'

for fileName in os.listdir(file_delete_dir):
    if os.path.isfile(os.path.join(file_delete_dir, fileName)):
        os.remove(os.path.join(file_delete_path, fileName))