import os
import sys

folder = str(sys.argv[1])
# folder = 'hi_hat'

count = 1

folder_path = f'data/_seperated/{folder}'
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for file_name in files:
    if count < 10:
        new_name = f'{folder}_00{count}.wav'
    elif count >= 10 and count < 100:
        new_name = f'{folder}_0{count}.wav'
    else:
        new_name = f'{folder}_{count}.wav'
    
    new_path = os.path.join(folder_path, new_name)
    os.rename(os.path.join(folder_path, file_name), new_path)

    count += 1