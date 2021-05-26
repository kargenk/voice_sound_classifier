import glob
import re

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import MelspDataset, make_datapath_list
from model import Classifier


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def dataclass_ratio(phase: str):
    human = 0
    non_human = 0
    path_list = make_datapath_list(phase)
    for path in path_list:
        is_human = True if re.search('basic', path, flags=re.IGNORECASE) else False
        if is_human:
            human += 1
        else:
            non_human += 1
    return human, non_human


def restore_model(path: str):
    model = Classifier()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def save_input_img(mspec: np.ndarray, title: str, save_path='./outputs/results/'):
    """モデルへの入力となるデータをグレースケール画像として保存する"""
    plt.figure(figsize=(4, 4))
    plt.imshow(mspec[0].squeeze().detach().cpu(), cmap='gray',
               norm=Normalize(vmin=-0.2, vmax=0.2))
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(f'{save_path}{title}.png')


if __name__ == '__main__':
    # 乱数のシード値固定
    SEED = 765
    fix_seed(SEED)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_list = make_datapath_list('train')
    train_dataset = MelspDataset(train_list)
    train_ratio = dataclass_ratio('train')
    print(f'train_data: {len(train_dataset)} = human: {train_ratio[0]} + not_human: {train_ratio[1]}')

    val_list = make_datapath_list('val')
    val_dataset = MelspDataset(val_list)
    train_ratio = dataclass_ratio('val')
    print(f'val_data: {len(train_dataset)} = human: {train_ratio[0]} + not_human: {train_ratio[1]}')

    # データローダの準備
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    model = restore_model('models/ep98_acc1.0000.pt')  # 学習済モデルの読み込み

    # for phase in ['train', 'val']:
    #     for inputs, labels in tqdm(dataloaders_dict[phase]):
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)

    #         with torch.set_grad_enabled(False):
    #             outputs = model(inputs.float())
    #             _, preds = torch.max(outputs, dim=1)

    # データの準備
    batch_iterator = iter(train_dataloader)
    inputs, labels = next(batch_iterator)
    print(inputs.size())
    print(labels)

    # 生のデータを描画
    for i in range(len(inputs)):
        # print(type(labels[i].item()))
        save_input_img(inputs[i], str(labels[i].item()) + '_' + str(i))
