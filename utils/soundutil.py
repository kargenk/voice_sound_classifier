from typing import Tuple

import librosa
from librosa.filters import mel as librosa_mel_fn
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def save_sample(file_path, sampling_rate, audio):
    """
    Helper function to save sample

    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 22050)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)


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
    # melsp_db = librosa.power_to_db(melsp, ref=np.max)
    img = librosa.display.specshow(melsp, hop_length=256, sr=sr, x_axis='time', y_axis='mel',
                                   ax=ax, cmap='magma')  # colorbarの一意化
    ax.set(title='Mel-frequency spectrogram')
    plt.colorbar(img, ax=ax, format='%2.0f dB')
    plt.show()


def wav_from_mspec(mspec, phase=None, hop_length=None, n_fft=None):
    if phase:
        inv = mspec * np.exp(1j * phase)  # 極形式から直交形式に変換(位相がある場合)
        rec = librosa.istft(inv, hop_length=hop_length, win_length=n_fft)
    else:
        rec = librosa.griffinlim(mspec, hop_length=hop_length, win_length=n_fft)  # Griffinlimで強度のみから位相を推定
    return rec


class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))  # db単位に変換
        return log_mel_spec


if __name__ == '__main__':
    wav_path = 'data/human/jsut_basic5000/BASIC5000_0001.wav'
    # wav_path = 'data/not_human/esc50/1-137-A-32.wav'
    data, sr = load_wav(wav_path, sr=24000)
    # melsp, S = calc_melsp(data)
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    print(S.shape)

    # メルスペクトログラムの描画
    show_melsp(S, sr)

    # # 描画
    # show_wave(data)
    # show_melsp(melsp, sr)

    # # 復元
    # sf.write('./test_origin_rec.wav', data, sr, subtype='PCM_24')
    # save_wav('./test_rec.wav', S, sr, hop_length=None, n_fft=None)
