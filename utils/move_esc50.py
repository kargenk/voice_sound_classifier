import os
import shutil
import sys

import pandas as pd

base_dir = os.path.abspath(os.path.join(os.pardir, 'data/ESC-50-master/'))
data_dir = os.path.join(base_dir, 'audio')
save_dir = os.path.join(os.path.join(os.pardir, 'data/human'))
meta_path = os.path.join(base_dir, 'meta/esc50.csv')

df = pd.read_csv(meta_path)

human_category = ['crying_baby', 'sneezing', 'clapping', 'breathing',
                  'coughing', 'footsteps', 'laughing', 'brushing_teeth',
                  'snoring', 'drinking_sipping']

for hc in human_category:
    wav_files = df[df.category == hc].filename.tolist()
    for wav in wav_files:
        save_path = os.path.join(save_dir, wav)
        try:
            shutil.move(os.path.join(data_dir, wav), save_path)
        except:
            pass
        print(f'[move] {save_path}')
