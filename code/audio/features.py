"""Audio front end shared by training and prediction.

- logmel(): 80-band log-mel spectrogram (the CNN input), 401 frames for a 4 s clip.
- handcrafted(): 88-dim vector of classic spoofing cues, including phase statistics
  (vocoders often reconstruct phase poorly even when the magnitude looks right).
"""
import io

import librosa
import numpy as np
import soundfile as sf

SR = 16000
SECONDS = 4
N_SAMPLES = SR * SECONDS
N_FFT = 512
HOP = 160
N_MELS = 80
N_MFCC = 20
N_PHASE_BANDS = 8
FEAT_DIM = 2 * N_MFCC + N_MFCC + 8 + 2 + 2 * N_PHASE_BANDS + 2  # 88


def fix_length(wave):
    """First 4 s of the clip; shorter clips are repeat-padded (zero padding would leak duration)."""
    if len(wave) < N_SAMPLES:
        wave = np.tile(wave, int(np.ceil(N_SAMPLES / max(len(wave), 1))))
    return wave[:N_SAMPLES].astype(np.float32)


def decode(data_or_path):
    """FLAC/WAV bytes or a file path -> 16 kHz mono float32, fixed 4 s."""
    if isinstance(data_or_path, (bytes, bytearray)):
        wave, sr = sf.read(io.BytesIO(data_or_path), dtype="float32", always_2d=True)
        wave = wave.mean(1)
    else:
        wave, sr = librosa.load(data_or_path, sr=None, mono=True)
    if sr != SR:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=SR)
    # ASVspoof 2019 shortcut: bonafide clips carry longer leading/trailing silence than spoofs.
    # Trim it so the model must learn from the speech itself, not from silence length.
    trimmed, _ = librosa.effects.trim(wave, top_db=30)
    if len(trimmed) >= SR // 2:
        wave = trimmed
    return fix_length(wave)


def _stft(wave):
    return librosa.stft(wave, n_fft=N_FFT, hop_length=HOP, window="hann")


def logmel(wave, S=None):
    S = _stft(wave) if S is None else S
    mel = librosa.feature.melspectrogram(S=np.abs(S) ** 2, sr=SR, n_mels=N_MELS)
    return np.log(mel + 1e-6).astype(np.float32)  # [80, 401]


def _wrap(x):
    return np.angle(np.exp(1j * x))


def phase_features(S):
    """Instantaneous-frequency deviation per band (mean |dev|, std) + group-delay stats."""
    mag, ph = np.abs(S), np.angle(S)
    k = np.arange(S.shape[0])[:, None]
    # Phase advance between frames minus what a stationary sinusoid at bin k would give.
    if_dev = _wrap(np.diff(ph, axis=1) - 2 * np.pi * k * HOP / N_FFT)
    bands = np.array_split(np.arange(S.shape[0]), N_PHASE_BANDS)
    a = np.abs(if_dev)
    out = []
    for b in bands:
        out += [a[b].mean(), a[b].std()]
    gd = -_wrap(np.diff(ph, axis=0))  # group delay along frequency
    wg = mag[1:] / (mag[1:].sum(0, keepdims=True) + 1e-9)
    gd_w = (np.abs(gd) * wg).sum(0)  # energy-weighted, so silent bins don't dominate
    out += [gd_w.mean(), gd_w.std()]
    return np.array(out, dtype=np.float32)


def handcrafted(wave, S=None):
    S = _stft(wave) if S is None else S
    mag = np.abs(S)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(
        librosa.feature.melspectrogram(S=mag ** 2, sr=SR, n_mels=N_MELS)), n_mfcc=N_MFCC)
    d = librosa.feature.delta(mfcc)
    spec = [librosa.feature.spectral_centroid(S=mag, sr=SR),
            librosa.feature.spectral_bandwidth(S=mag, sr=SR),
            librosa.feature.spectral_rolloff(S=mag, sr=SR),
            librosa.feature.spectral_flatness(S=mag)]
    zcr = librosa.feature.zero_crossing_rate(wave, frame_length=N_FFT, hop_length=HOP)
    parts = [mfcc.mean(1), mfcc.std(1),
             np.abs(d).mean(1),  # mean |delta|: the plain mean of a delta is ~0 by construction
             np.array([f(s) for s in spec for f in (np.mean, np.std)]),
             np.array([zcr.mean(), zcr.std()]),
             phase_features(S)]
    return np.concatenate(parts).astype(np.float32)


def wave_to_inputs(wave):
    """One STFT shared by both outputs."""
    S = _stft(wave)
    return logmel(wave, S), handcrafted(wave, S)
