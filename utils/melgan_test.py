import librosa
import librosa.display
import numpy as np
import soundfile as sf
import torch

from soundutil import load_wav, show_melsp, save_sample, Audio2Mel

if __name__ == '__main__':
    wav_path = 'data/human/jsut_basic5000/BASIC5000_0001.wav'
    fft = Audio2Mel()
    data, sr = load_wav(wav_path, sr=24000)
    data = torch.from_numpy(data).float().unsqueeze(0)
    log_melsp = fft(data.unsqueeze(0))
    show_melsp(log_melsp.squeeze().numpy(), sr)

    # MelGANを用いてでメルスペクトログラムを音声に復元
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    audio = vocoder.inverse(log_melsp).squeeze()
    print(min(audio), max(audio))

    # wav形式で保存
    sf.write('rec_sf.wav', audio, sr, subtype='PCM_16')
    save_sample('./reconstruct.wav', sampling_rate=sr, audio=audio)
