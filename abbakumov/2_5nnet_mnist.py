#  Активируем библиотеки
import numpy as np
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras import utils
import datetime
import random

#  Задаем параметры сети
batch_size = 128
nr_classes = 10
nr_iterations = 5

#  Читаем данные
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#  картинку вытягиваем в столбец
#  Информация о взаимном расположении теряется (не совсем)
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

#  Уточняем тип данных Нормируем входные значения
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

#  Делаем 10 бинарных столбцов (так как 10 цифр)
Y_train = utils.to_categorical(y_train, nr_classes)
Y_test = utils.to_categorical(y_test, nr_classes)

#  Описываем сеть. Один внутренний слой
model = Sequential()
model.add(Dense(128, input_shape=(784,)))  # Наверное здесь линейная функция активации по умолчанию
model.add(Activation('relu'))  # relu сделана отдельным слоем, может это нужно для Dropout?
# model.add(Dropout(0.5))  # Ухудшает немного
model.add(Dense(10))
model.add(Activation('softmax'))

#  Проверяем себя
model.summary()

#  Определяем параметры обучения
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

seed = random.randrange(1000)  # seed = random.randrange(1000); np.random.seed(a) # тоже самое вроде бы
tensorflow.random.set_seed(seed=1)  # Результат не воспроизводится почему-то

# workers=2, use_multiprocessing=True Это для распределенной работы видимо, все ядра и так работают
net_res_1 = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nr_iterations,
                      verbose=1, validation_data=(X_test, Y_test))

test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Loss:", test_loss)
print("Accuracy:", test_acc)  # 97.5%
