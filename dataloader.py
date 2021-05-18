import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def make_datapath_list(phase: str = 'train') -> list:
    """
    trainかvalを指定してデータへのパスのリストを返す関数.

    Args:
        phase (str, optional): 訓練かテストかのフラグ. Defaults to 'train'.

    Returns:
        list: ファイルのパスリスト
    """
    root_dir = './data/processed/'
    target_path = os.path.join(root_dir, phase, '*.npy')
    print(f'search: {target_path}')

    path_list = []

    # サブディレクトリまで探査
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


class MelspDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        melsp_path = self.file_list[index]
        data = np.load(melsp_path)
        melsp = torch.from_numpy(data).unsqueeze(0)

        is_human = True if re.search('basic', melsp_path, flags=re.IGNORECASE) else False
        if is_human:
            label = 1
        else:
            label = 0

        return melsp, label


if __name__ == '__main__':
    train_list = make_datapath_list('train')
    train_dataset = MelspDataset(train_list)
    print(len(train_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    batch_iterator = iter(train_dataloader)
    inputs, labels = next(batch_iterator)
    print(inputs.size())
    print(labels)
