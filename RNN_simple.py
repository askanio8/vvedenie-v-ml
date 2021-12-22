import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # уровень отображения предупреждений системы

import numpy as np
import re

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer

with open('train_data_true', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '')  # убираем первый невидимый символ
    text = re.sub(r'[^А-я ]', '', text)  # заменяем все символы кроме кириллицы на пустые символы


# парсим текст, как последовательность символов
num_characters = 34  # 33 буквы + пробел
# токенизируем на уровне символов
tokenizer = Tokenizer(num_words=num_characters, # количество самых встречаемых слов(символов) по частоте
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', # фильтр по умолчанию
                      char_level=True,  # ннтересуют символы а не слова
                      lower=True, split=' ',
                      oov_token=None  # можно задать замену слов не из num_words
                      )
tokenizer.fit_on_texts([text])  # формируем токены на основе частотности в нашем тексте
print(tokenizer.word_index)  # словарь, где каждому символу соответствует число


inp_chars = 5
data = tokenizer.texts_to_matrix(text)  # преобразуем исходный текст в массив one-hot encoding
# индекс превого Y
n = data.shape[0] - inp_chars  # так как мы предсказываем по трем символам - четвертый

# X - трехмерная разреженная матрица num_characters X inp_chars X i-inp_chars
X = np.array([data[i:i + inp_chars, :] for i in range(n)])
# Y - двумерная разреженная матрица
Y = data[inp_chars:]  # предсказание следующего символа

print(data.shape)

model = Sequential()  # модель получает на вход двумерную разреженную матрицу(с учетом размера батча трехмерную)
model.add(Input((inp_chars,
                 num_characters)))  # при тренировке в рекуррентные модели keras подается сразу вся последовательность, поэтому в input теперь два числа. 1-длина последовательности, 2-размер OHE
model.add(SimpleRNN(128, activation='tanh'))  # рекуррентный слой на 500 нейронов. tanh - стандартный выбор в RNN
model.add(Dense(num_characters, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X, Y, batch_size=32, epochs=100)


def buildPhrase(inp_str, str_len=50):
    for i in range(str_len):
        x = []
        for j in range(i, i + inp_chars):
            x.append(tokenizer.texts_to_matrix(inp_str[j]))  # преобразуем символы в One-Hot-encoding

        x = np.array(x)
        inp = x.reshape(1, inp_chars, num_characters)

        pred = model.predict(inp)  # предсказываем OHE четвертого символа
        d = tokenizer.index_word[pred.argmax(axis=1)[0]]  # получаем ответ в символьном представлении

        inp_str += d  # дописываем строку

    return inp_str


res = buildPhrase("мысли")
print(res)