"""
model_loader.py — загрузка модели и предобработка аудио
Совместим с нейросетью обученной через adwa.py (librosa, sr=16000, n_mels=128)
"""

import os, io, glob, logging
import numpy as np

log = logging.getLogger(__name__)

# ── Параметры ТОЧНО как в adwa.py ──────────────────────────────────
SR        = 16000   # sr=16000 в librosa.feature.melspectrogram
N_FFT     = 1024
HOP       = 512
N_MELS    = 128     # n_mels=128
# Реальный input shape модели: (128, 157, 1)
# — librosa с этими параметрами даёт ~157 фреймов на аудио из датасета

# Глобальный кеш
_model      = None
_model_path = None
_mean       = None   # нормализация из train
_std        = None


def _find_model(base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    # Приоритеты — добавлено audio_resnet_model.h5
    for name in ['audio_resnet_model.h5', 'model.h5', 'alien_model.h5',
                 'best_model.h5', 'best_encoder.h5']:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p
    h5 = glob.glob(os.path.join(base_dir, '*.h5'))
    if h5:
        return h5[0]
    out = os.path.join(base_dir, 'outputs')
    if os.path.isdir(out):
        h5 = glob.glob(os.path.join(out, '*.h5'))
        if h5:
            return h5[0]
    return None


def load_model(path=None):
    global _model, _model_path
    if _model is not None:
        return _model, _model_path
    if path is None:
        path = _find_model()
    if path is None:
        log.warning('Файл .h5 не найден — случайные предсказания')
        return None, None
    try:
        from tensorflow import keras
        _model = keras.models.load_model(path, compile=False)
        _model_path = path
        log.info(f'Модель загружена: {path}')
        log.info(f'Input:  {_model.input_shape}')
        log.info(f'Output: {_model.output_shape}')
        return _model, _model_path
    except Exception as e:
        log.error(f'Ошибка загрузки {path}: {e}')
        return None, None


# ── Конвертация аудио ТОЧНО как в adwa.py ──────────────────────────
def _clean_audio(audio):
    """Повторяет clean_audio() из adwa.py."""
    audio = np.asarray(audio)
    audio = np.squeeze(audio)
    audio = audio.flatten()
    return audio.astype(np.float32)


def _audio_to_mel(audio):
    """Повторяет audio_to_mel() из adwa.py — librosa, sr=16000, n_mels=128."""
    import librosa
    audio = _clean_audio(audio)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def _audio_to_mel_scipy(audio):
    """Запасной вариант без librosa — через scipy."""
    from scipy.io import wavfile
    from scipy import signal as sp

    audio = _clean_audio(audio)
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    # Мел-фильтрбанк
    n_freqs = N_FFT // 2 + 1
    mel_max = 2595 * np.log10(1 + (SR / 2) / 700)
    mel_pts = np.linspace(0, mel_max, N_MELS + 2)
    hz_pts  = 700 * (10 ** (mel_pts / 2595) - 1)
    bins    = np.clip(np.floor(hz_pts * N_FFT / SR).astype(int), 0, n_freqs - 1)
    fbank   = np.zeros((N_MELS, n_freqs))
    for m in range(1, N_MELS + 1):
        l, c, r = bins[m-1], bins[m], bins[m+1]
        for k in range(l, c):
            if c > l: fbank[m-1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c: fbank[m-1, k] = (r - k) / (r - c)

    _, _, spec = sp.spectrogram(audio, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    mel  = fbank @ spec
    mel_db = 10 * np.log10(mel + 1e-10)
    mel_db -= mel_db.max()
    return mel_db


def _wav_bytes_to_audio(wav_bytes):
    """Байты wav → float32 массив."""
    from scipy.io import wavfile
    from scipy import signal as sp
    try:
        rate, audio = wavfile.read(io.BytesIO(bytes(wav_bytes)))
    except Exception:
        return np.zeros(SR * 3, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    # Ресэмплинг → SR=16000
    if rate != SR:
        from scipy import signal as sp2
        audio = sp2.resample(audio, int(len(audio) * SR / rate)).astype(np.float32)
    return audio


def _process_batch(wav_array):
    """wav_array → numpy (N, H, W, 1) нормализованных спектрограмм."""
    global _mean, _std

    # Конвертируем
    try:
        import librosa
        specs = [_audio_to_mel(_wav_bytes_to_audio(wb)) for wb in wav_array]
        log.info('Предобработка через librosa')
    except ImportError:
        log.warning('librosa не установлен — используем scipy')
        specs = [_audio_to_mel_scipy(_wav_bytes_to_audio(wb)) for wb in wav_array]

    # Выравниваем ширину (librosa может дать разную длину)
    T = max(s.shape[1] for s in specs)
    padded = []
    for s in specs:
        if s.shape[1] < T:
            s = np.pad(s, ((0, 0), (0, T - s.shape[1])))
        else:
            s = s[:, :T]
        padded.append(s)

    X = np.array(padded, dtype=np.float32)  # (N, 128, T)

    # Нормализация — как в adwa.py: (x - mean) / std по всему train
    # Если нет сохранённых mean/std — нормализуем по текущему батчу
    if _mean is not None and _std is not None:
        X = (X - _mean) / (_std + 1e-9)
    else:
        m, s = X.mean(), X.std()
        if s > 0:
            X = (X - m) / s

    X = X[..., np.newaxis]  # (N, 128, T, 1)
    return X


def predict_from_wav_array(wav_array, model=None):
    """
    Принимает массив wav-байт → предсказания (N, num_classes).
    """
    if model is None:
        model, _ = load_model()

    X = _process_batch(wav_array)
    log.info(f'Спектрограммы: {X.shape}')

    if model is None:
        num_classes = 802
        rng = np.random.default_rng(42)
        preds = rng.dirichlet(np.ones(num_classes), size=len(wav_array)).astype(np.float32)
        log.warning('Модель не найдена — случайные предсказания')
        return preds

    # Подгоняем ширину спектрограммы под ожидаемый input модели
    model_h = model.input_shape[1]
    model_w = model.input_shape[2]
    _, h, w, _ = X.shape

    if h != model_h or w != model_w:
        log.info(f'Ресайз спектрограмм: ({h},{w}) → ({model_h},{model_w})')
        from scipy.ndimage import zoom
        resized = []
        for s in X:
            zy = model_h / h
            zx = model_w / w
            resized.append(zoom(s[:, :, 0], (zy, zx))[:, :, np.newaxis])
        X = np.array(resized, dtype=np.float32)
    else:
        log.info(f'Размер спектрограмм совпадает с моделью: ({h},{w})')

    preds = model.predict(X, verbose=0)
    log.info(f'Предсказания: {preds.shape}, max_conf={float(preds.max()):.4f}')
    return preds


def save_normalization(X_train):
    """Вызови после обучения чтобы сохранить mean/std для инференса."""
    global _mean, _std
    _mean = float(X_train.mean())
    _std  = float(X_train.std())
    np.save('norm_params.npy', np.array([_mean, _std]))
    log.info(f'Сохранены параметры нормализации: mean={_mean:.4f}, std={_std:.4f}')


def load_normalization(path='norm_params.npy'):
    """Загружает mean/std если они были сохранены."""
    global _mean, _std
    if os.path.exists(path):
        params = np.load(path)
        _mean, _std = float(params[0]), float(params[1])
        log.info(f'Нормализация загружена: mean={_mean:.4f}, std={_std:.4f}')
        return True
    return False

# Пробуем загрузить нормализацию при импорте
load_normalization()