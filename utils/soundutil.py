from typing import Tuple

import librosa
import librosa.display
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def load_wav(wav_path: str, sr: int = None, info: bool = False) -> Tuple[np.ndarray, int]:
    """
    wavファイルを読み込む関数.

    Args:
        wav_path (str): 音声データのパス
        sr (int): サンプリング周波数
        info (bool, optional): 情報を表示するかのフラグ. Defaults to False.

    Returns:
        Tuple[np.ndarry, int]: データとサンプリング周波数
    """
    data, sr = librosa.load(wav_path, sr)
    data = data.astype(np.float)  # change to ndarray
    if info:
        print(f'sampling rate: {sr}')
        print(f'wave size: {data.shape}')
        print(f'time: {len(data) // sr} s')
    return data, sr


def calc_melsp(x: np.ndarray, n_fft: int = 1024, hop_length: int = 128, info: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    メルスペクトログラムを計算する関数.

    Args:
        x (np.ndarray): 音声データ
        n_fft (int, optional): 窓サイズ. Defaults to 1024.
        hop_length (int, optional): 移動幅. Defaults to 128.
        info (bool, optional): 情報を表示するかのフラグ. Defaults to False.

    Returns:
        np.ndarray: メルスペクトログラム
        np.ndarray: 強度の行列
    """
    stft = np.abs(librosa.stft(x)) ** 2
    S, phase = librosa.magphase(stft)  # 複素数を強度と位相に分解
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
    if info:
        print(f'melsp size: {melsp.shape}')
        print(f'S size: {S.shape}')
    return melsp, S


def show_wave(x: np.ndarray) -> None:
    plt.plot(x)
    plt.show()


def show_melsp(melsp: np.ndarray, sr: int) -> None:
    fig, ax = plt.subplots()
    img = librosa.display.specshow(melsp, sr=sr, x_axis='time', y_axis='log',
                                   ax=ax, cmap='magma', norm=Normalize(vmin=-2, vmax=2))  # colorbarの一意化
    plt.colorbar(img, ax=ax, format='%2.0f dB')
    plt.show()


def wav_from_mspec(mspec, phase=None, hop_length=None, n_fft=None):
    if phase:
        inv = mspec * np.exp(1j * phase)  # 極形式から直交形式に変換(位相がある場合)
        rec = librosa.istft(inv, hop_length=hop_length, win_length=n_fft)
    else:
        rec = librosa.griffinlim(mspec, hop_length=hop_length, win_length=n_fft)  # Griffinlimで強度のみから位相を推定
    return rec


def save_wav(path, mspec_power, sr, hop_length=None, n_fft=None):
    wav = wav_from_mspec(mspec_power, hop_length=hop_length, n_fft=n_fft)
    sf.write(path, wav, sr, subtype='PCM_24')


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


if __name__ == '__main__':
    wav_path = 'data/human/jsut_basic5000/BASIC5000_0001.wav'
    # wav_path = 'data/not_human/esc50/1-137-A-32.wav'
    data, sr = load_wav(wav_path, sr=None)
    melsp, S = calc_melsp(data)

    # 描画
    show_wave(data)
    show_melsp(melsp, sr)

    # 復元
    sf.write('./test_origin_rec.wav', data, sr, subtype='PCM_24')
    save_wav('./test_rec.wav', S, sr, hop_length=None, n_fft=None)
