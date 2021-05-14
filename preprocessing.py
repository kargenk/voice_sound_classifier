import glob
import os

import numpy as np
from sklearn.model_selection import train_test_split

import utils.soundutil as util

FRAMES = 128


def save_np_data(x: str, y: int, aug=None, rate=None, flag: str = '') -> None:
    """
    データ拡張してメルスペクトログラムを保存する関数.

    Args:
        x (str): 音声データのパス
        y (int): クラスラベル
        aug ([type], optional): データ拡張の関数. Defaults to None.
        rates ([type], optional): データ拡張関数の係数. Defaults to None.
        flag (str, optional): 訓練データかラベルかのフラグ. Defaults to ''.
    """
    sr = 24000
    _x, sr = util.load_wav(x, sr)
    if aug is not None:
        _x = aug(x=_x, rate=rate)
    melsp, S = util.calc_melsp(_x)
    filename = os.path.join('data/processed/', flag, os.path.basename(x[:-4]) + '.npz')
    np.savez(filename, x=S, y=y, melsp=melsp)
    print(f'[SAVE]: {filename}')

    # # 正しく保存できているか確認
    # _S = np.load(filename)
    # print(S == _S['x'])


if __name__ == '__main__':
    human_wavs = glob.glob('./data/human/*/*.wav')
    human_class = np.ones(len(human_wavs))
    not_human_wavs = glob.glob('./data/not_human/*/*.wav')
    not_human_class = np.zeros(len(not_human_wavs))
    # 訓練データとラベルをまとめる
    x = human_wavs + not_human_wavs
    y = np.concatenate([human_class, not_human_class])

    # 訓練データとテストデータに分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42,
                                                        test_size=0.25, stratify=y)
    print(f'x_train: {len(x_train)}\ny_train: {len(y_train)}\nx_test: {len(x_test)}\ny_test: {len(y_test)}\n')

    for x_t, y_t in zip(x_train, y_train):
        save_np_data(x_t, y_t, flag='train')
    for x_v, y_v in zip(x_test, y_test):
        save_np_data(x_v, y_v, flag='val')
