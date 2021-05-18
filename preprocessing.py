import glob
import os

import numpy as np
from sklearn.model_selection import train_test_split

import utils.soundutil as util

FRAMES = 128


def add_white_noise(x: np.ndarray, rate: float = 0.002) -> np.ndarray:
    """
    ホワイトノイズを追加する関数.

    Args:
        x (np.ndarray): 音声データ
        rate (float, optional): ノイズの大きさを調節する係数. Defaults to 0.002.

    Returns:
        np.ndarray: ホワイトノイズを付加した音声データ
    """
    return x + rate * np.random.randn(len(x))


def shift_sound(x: np.ndarray, rate: int = 2) -> np.ndarray:
    """
    時間軸上でシフトさせる関数.

    Args:
        x (np.ndarray): 音声データ
        rate (int, optional): シフトする割合. Defaults to 2.

    Returns:
        np.ndarray: シフト後の音声データ
    """
    return np.roll(x, int(len(x) // rate))


def stretch_sound(x: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """
    音声を伸縮させる関数.

    Args:
        x (np.ndarray): 音声データ
        rate (float, optional): 伸縮させる割合. Defaults to 1.1.

    Returns:
        np.ndarray: 伸縮後の音声
    """
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)

    # 長すぎたら切り取り，短すぎたらパディング
    if len(x) > input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), 'constant')


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
    processed_dir = 'data/processed/'
    _x, sr = util.load_wav(x, sr)
    if aug is not None:
        _x = aug(x=_x, rate=rate)
    melsp, S = util.calc_melsp(_x)

    # 全データはnpzで保存
    filename = os.path.join(processed_dir, flag, os.path.basename(x[:-4]) + '.npz')
    if not os.path.exists(filename):
        np.savez(filename, x=S, y=y, melsp=melsp)
        print(f'[SAVE]: {filename}')

    # # 正しく保存できているか確認
    # _S = np.load(filename)
    # print(S == _S['x'])

    # 128フレームずつに切ってnpyで保存，現状余りは捨てている
    for start_idx in range(0, S.shape[1] - FRAMES + 1, FRAMES):
        one_audio_seg = S[:, start_idx: start_idx + FRAMES]

        if one_audio_seg.shape[1] == FRAMES:
            file_path = f'{filename[:-4]}_{start_idx}'
            if not os.path.exists(file_path + '.npy'):
                np.save(file_path, one_audio_seg)
                print(f'[SAVE]: {file_path}.npy')


if __name__ == '__main__':
    human_wavs = glob.glob('./data/human/*/*.wav')
    human_class = np.ones(len(human_wavs), dtype=np.int8)
    not_human_wavs = glob.glob('./data/not_human/*/*.wav')
    not_human_class = np.zeros(len(not_human_wavs), dtype=np.int8)
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
