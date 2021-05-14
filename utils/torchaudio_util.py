from IPython.display import Audio, display
import matplotlib.pyplot as plt
import torch
import torchaudio


def print_metadata(metadata, src: str = None):
    """
    音声のメタデータを出力する関数．

    Args:
        metadata (torchaudio.backend.common.AudioMetaData): torchaudio.info()で得られるメタデータ
        src (str, optional): 音源ファイルのパス. Defaults to None.
    """
    if src:
        print('-' * 10)
        print(f'Source: {src}')
        print('-' * 10)
    print(f' - sample_rate: {metadata.sample_rate}')
    print(f' - num_channels: {metadata.num_channels}')
    print(f' - num_frames: {metadata.num_frames}')
    print(f' - bits_per_sample: {metadata.bits_per_sample}')
    print(f' - encoding: {metadata.encoding}')
    print()


def print_stats(waveform: torch.Tensor, sample_rate: int = None, src: str = None):
    """
    音声データの統計情報を出力する関数.

    Args:
        waveform (torch.Tensor): torchaudio.loadで読み込んだ音声データのテンソル
        sample_rate (int, optional): サンプリング周波数. Defaults to None.
        src (str, optional): 音声ファイルのパス. Defaults to None.
    """
    if src:
        print('-' * 10)
        print(f'Source: {src}')
        print('-' * 10)
    if sample_rate:
        print(f'Sample Rate: {sample_rate}')
    print(f'Shape: {tuple(waveform.shape)}')
    print(f'Dtype: {waveform.dtype}')
    print(f' - Max:     {waveform.max().item():6.3f}')
    print(f' - Min:     {waveform.min().item():6.3f}')
    print(f' - Mean:    {waveform.mean().item():6.3f}')
    print(f' - Std Dev: {waveform.std().item():6.3f}')
    print()
    print(waveform)
    print()


def plot_waveform(waveform: torch.Tensor, sample_rate: int, title: str = 'Waveform', xlim: float = None, ylim: float = None):
    """
    波形データを描画する関数.py

    Args:
        waveform (torch.Tensor): 音声データのテンソル
        sample_rate (int): サンプリング周波数
        title (str, optional): グラフのタイトル. Defaults to 'Waveform'.
        xlim (float, optional): 開始秒数. Defaults to None.
        ylim (float, optional): 表示する下限値. Defaults to None.
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    # plt.show(block=False)  # for Jupyter Notebook
    plt.show()


def plot_specgram(waveform: torch.Tensor, sample_rate: int, title: str = 'Spectrogram', xlim: float = None):
    """
    スペクトログラムを描画する関数.

    Args:
        waveform (torch.Tensor): 音声データのテンソル
        sample_rate (int): サンプリング周波数
        title (str, optional): グラフのタイトル. Defaults to 'Spectrogram'.
        xlim (float, optional): 開始秒数. Defaults to None.
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    # plt.show(block=False)  # for Jupyter Notebook
    plt.show()


def play_audio(waveform: torch.Tensor, sample_rate: int):
    """
    Jupyter Notebook上で再生ウィジェットを作成する関数.

    Args:
        waveform (torch.Tensor): 音声データのテンソル
        sample_rate (int): サンプリング周波数

    Raises:
        ValueError: 2ch以上のテンソルはサポート外
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


if __name__ == '__main__':
    wav_path = 'data/human/jsut_basic5000/BASIC5000_0001.wav'

    # # メタデータ
    # metadata = torchaudio.info(wav_path)
    # util.print_metadata(metadata, src=wav_path)

    # 音声データの読み込み
    waveform, sr = torchaudio.load(wav_path)
    print_stats(waveform, sr)    # 統計情報

    # 波形とスペクトログラムの描画
    plot_waveform(waveform, sr)  # 波形を描画
    plot_specgram(waveform, sr)  # スペクトログラムを描画
