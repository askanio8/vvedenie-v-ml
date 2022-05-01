import timit_utils as tu
import timit_utils.audio_utils as au
import timit_utils.drawing_utils as du

import os
import librosa                  #?
import numpy as np
from tqdm import tqdm           #?

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import torchaudio

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import soundfile as sf

import IPython.display  # для воспроизведения звука
_TIMIT_PATH = 'data/lisa/data/timit/raw/TIMIT'


# пробуем прочитать файл
data, sr = librosa.load('sample_f.mp3', 16000)  # нужен ffmpeg
print(data[:10])
print(type(data))
print(data.shape)  # 16000 * секунд
# рисуем звук
plt.plot(data)
plt.show()

IPython.display.Audio(data, rate=16000)  # в pycharm не проигрывает, но и ошибку не пишет

_sr, _data = sf.read('tmp.wav')  # запасной вариант чтения файлов?

# теперь читаем wav
amplitudes, sample_rate = librosa.load('tmp.wav')
print(f"{len(amplitudes)} points, {len(amplitudes) / sample_rate} sec, sr {sample_rate}")

# режем массив на куски с перекрытием hop_length
def slice_into_frames(amplitudes, window_length, hop_length):
    return librosa.core.spectrum.util.frame(
        np.pad(amplitudes, int(window_length // 2), mode='reflect'),  # дополняем массив с краев на ширину пол окна
        frame_length=window_length, hop_length=hop_length)
    # выход: [window_length, num_windows]


def get_STFT(amplitudes, window_length, hop_length):
    """ Compute short-time Fourier Transform """
    # разбиваем амплитуды на пересекающиеся фреймы
    # нелогичен порядок размерностей [window_length, num_frames] а не наоборот!
    frames = slice_into_frames(amplitudes, window_length, hop_length)

    # получаем веса для Фурье, float[window_length]
    # окно ханна
    fft_weights = librosa.core.spectrum.get_window('hann', window_length, fftbins=True)

    # применяем преобразование Фурье
    stft = np.fft.rfft(frames * fft_weights[:, None], axis=0)
    return stft


def get_melspectrogram(amplitudes, sample_rate=22050, n_mels=128,
                       window_length=2048, hop_length=512, fmin=1, fmax=8192):
    """
    Implement mel-spectrogram as described above.
    :param amplitudes: float [num_amplitudes]
    :param sample_rate: число отсчетов каждую секунду
    :param n_mels: число каналов спектрограммы
    :param window_length: параметр размера окна для Фурье
    :param hop_length: размер пересечения
    :param f_min: мин частота
    :param f_max: макс частота
    :returns: мел-scaled спектрограмма [n_mels, duration]
    """
    # Шаг 1
    stft = get_STFT(amplitudes, window_length, hop_length)
    assert stft.shape == (window_length // 2 + 1, len(amplitudes) // 512 + 1)

    # Шаг 2 странное преобразование 'избавляемся от комплексной составляющей'
    spectrogram = np.abs(stft ** 2)


    # Шаг 3  берем диапазон слышимости человека
    mel_basis = librosa.filters.mel(sample_rate, n_fft=window_length,
                                    n_mels=n_mels, fmin=fmin, fmax=fmax)
    # ^-- matrix [n_mels, window_length / 2 + 1]

    mel_spectrogram = np.dot(mel_basis, spectrogram)
    assert mel_spectrogram.shape == (n_mels, len(amplitudes) // 512 + 1)

    return mel_spectrogram


amplitudes1, _  = librosa.load('sample_f.mp3', sr=16000)
amplitudes2, _  = librosa.load('sample_f.mp3', sr=16000)

np.allclose(amplitudes[0], amplitudes1)  # сравниваем результаты загрузки wav и mp3

# добываем мел-спектрограмму с помощью librosa
ref1 = librosa.feature.melspectrogram(amplitudes2, sr=sample_rate, n_mels=128, fmin=1, fmax=8192)
# сравниваем мел-спектрограммы сделанные своей ф-ей и от librosa
assert np.allclose(get_melspectrogram(amplitudes2), ref1, rtol=1e-4, atol=1e-4)

# добываем мел-спектрограмму с помощью torchaudio
# torchaudio с pycharm не дружит чтоли.
#amplitudes, sr = torchaudio.load('sample_f.mp3', sample_rate=16000)
#ref2 = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, f_min=1, n_mels=128, f_max=8192)(amplitudes)

#assert np.allclose(ref2, ref1, rtol=1e-4, atol=1e-4)


# рисуем мел-спектрограмму
plt.figure(figsize=[16, 4])
plt.title('name_file'); plt.xlabel('Time'); plt.ylabel('Frecuency')
plt.imshow(get_melspectrogram(amplitudes1).transpose(), cmap='rainbow')
plt.show()


# теперь используем датасет TIMIT
DATA_PATH = 'data/lisa/data/timit/raw/TIMIT'
corpus = tu.Corpus(DATA_PATH)  # передаем адрес данных
sentence = corpus.train.sentences_by_phone_df('aa').sentence[0]  # берем одно из высказываний
du.DrawVerticalPanels([du.AudioPanel(sentence.raw_audio, show_x_axis=True),  # рисуем график звука
                       du.WordsPanel(sentence.words_df, sentence.raw_audio.shape[0], show_x_axis=True),  # слова
                       du.PhonesPanel(sentence.phones_df, sentence.raw_audio.shape[0])  # фонемы
                      ])
plt.show()


class timit_dataloader:
    def __init__(self, data_path='./data/TIMIT', train_mode=True, age_mode=False):
        self.doc_file_path = os.path.join(data_path, 'DOC', 'SPKRINFO.TXT')
        self.corpus = tu.Corpus(data_path)
        with open(self.doc_file_path) as f:
            self.id_sex_dict = dict([(tmp.split(' ')[0], tmp.split(' ')[2]) for tmp in f.readlines()[39:]])
        with open(self.doc_file_path) as f:
            self.id_age_dict = dict(
                [(tmp.split(' ')[0], 86 - int(tmp.split('  ')[5].split('/')[-1].replace('??', '50'))) \
                 for tmp in f.readlines()[39:]])
        # print(self.id_age_dict)
        if train_mode:
            self.trainset = self.create_dataset('train', age_mode=age_mode)
            self.validset = self.create_dataset('valid', age_mode=age_mode)
        self.testset = self.create_dataset('test', age_mode=age_mode)

    def return_sex(self, id):
        return self.id_sex_dict[id]

    def return_age(self, id):
        return self.id_age_dict[id]

    def return_data(self):
        return self.trainset, self.validset, self.testset

    def return_test(self):
        return self.testset

    def create_dataset(self, mode, age_mode=False):
        global people
        assert mode in ['train', 'valid', 'test']
        if mode == 'train':
            people = [self.corpus.train.person_by_index(i) for i in range(350)]
        if mode == 'valid':
            people = [self.corpus.train.person_by_index(i) for i in range(350, 400)]
        if mode == 'test':
            people = [self.corpus.test.person_by_index(i) for i in range(150)]
        spectrograms_and_targets = []
        if age_mode:
            for person in tqdm(people):
                try:
                    target = self.return_age(person.name)
                    for i in range(len(person.sentences)):
                        spectrograms_and_targets.append(
                            self.preprocess_sample(person.sentence_by_index(i).raw_audio, target, age_mode=True))
                except:
                    print(person.name, target)
        else:
            for person in tqdm(people):
                target = self.return_sex(person.name)
                for i in range(len(person.sentences)):
                    spectrograms_and_targets.append(
                        self.preprocess_sample(person.sentence_by_index(i).raw_audio, target))

        X, y = map(np.stack, zip(*spectrograms_and_targets))
        X = X.transpose([0, 2, 1])  # to [batch, time, channels]
        return X, y

    @staticmethod
    def spec_to_image(spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled

    @staticmethod
    def clasterize_by_age(age):
        if age < 25:
            return 0
        if 25 < age < 40:
            return 0.5
        if age > 40:
            return 1

    def preprocess_sample(self, amplitudes, target, age_mode=False, sr=16000, max_length=150):
        spectrogram = librosa.feature.melspectrogram(amplitudes, sr=sr, n_mels=128, fmin=1, fmax=8192)[:, :max_length]
        spectrogram = np.pad(spectrogram, [[0, 0], [0, max(0, max_length - spectrogram.shape[1])]], mode='constant')
        if age_mode:
            # target = self.clasterize_by_age(target)
            target = target/80
        else:
            target = 0 if target == 'F' else 1
        # print(np.array(self.spec_to_image(np.float32(spectrogram))).shape)
        return self.spec_to_image(np.float32(spectrogram)), target

    def preprocess_sample_inference(self, amplitudes, sr=16000, max_length=150, device='cpu'):
        spectrogram = librosa.feature.melspectrogram(amplitudes, sr=sr, n_mels=128, fmin=1, fmax=8192)[:, :max_length]
        spectrogram = np.pad(spectrogram, [[0, 0], [0, max(0, max_length - spectrogram.shape[1])]], mode='constant')
        spectrogram = np.array([self.spec_to_image(np.float32(spectrogram))]).transpose([0, 2, 1])

        return t.tensor(spectrogram, dtype=t.float).to(device, non_blocking=True)


class dataloader:
    def __init__(self, spectrograms, targets):
        self.data = list(zip(spectrograms, targets))

    def next_batch(self, batch_size, device):
        indices = np.random.randint(len(self.data), size=batch_size)

        input = [self.data[i] for i in indices]

        source = [line[0] for line in input]
        target = [line[1] for line in input]

        return self.torch_batch(source, target, device)

    @staticmethod
    def torch_batch(source, target, device):
        return tuple(
            [
                t.tensor(val, dtype=t.float).to(device, non_blocking=True)
                for val in [source, target]
            ]
        )

# дальше сверточная сеть на pytorch
