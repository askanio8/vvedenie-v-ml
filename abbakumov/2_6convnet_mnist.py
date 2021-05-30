from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
import tensorflow.keras
from sklearn.model_selection import train_test_split
from sklearn import  metrics
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import tensorflow as tf


# Читаем данные из файла. Здесь не весь MNIST!!!!!!!!!!!!!!!!!!!!
train = pd.read_csv("train.csv")

# Разделяем предикторы и отклик
Y = train['label']
X = train.drop(['label'], axis=1)

# Разделяем на обучающую выборку и выборку валидации
x_train, x_val, y_train, y_val = train_test_split(X.values, Y.values, test_size=0.10, random_state=42)

# параметры сети, чтобы их было удобно менять
batch_size = 64
num_classes = 10
epochs = 5

# размерность картинки
img_rows, img_cols = 28, 28

# преобразование обучающей выборки
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_train /= 255

# преобразование выборки валидации
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
x_val = x_val.astype('float32')
x_val /= 255

#  преобразование тестовой выборки
#  Xtest = test.values
#  Xtest = Xtest.reshape(Xtest.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)
# преобразование отклика в 10 бинарных перменных
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_val = tensorflow.keras.utils.to_categorical(y_val, num_classes)

model = Sequential()
# первый сверточный слой
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# второй сверточный слой
model.add(Conv2D(64, (3, 3), activation='relu'))
# слой Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))  # выбраем из квадрата максимальное значение, таким образом повышаем
# контраст и уменьшаем размеры в 4 раза(в случае pool_size=(2, 2))
model.add(Dropout(0.25))  # слой dropout Не модифицируем веса части нейронов
model.add(Flatten())  # растягиваем в вектор # вытягиваем матрицу Х в линию перед обычным слоем
# первый слой анализа
model.add(Dense(128, activation='relu'))
# слой dropout
model.add(Dropout(0.5))
# второй слой анализа
model.add(Dense(num_classes, activation='softmax'))

# определяемся с обучением # можно Adadelta можно adam говорил Аббакумов, но здесь adam намного лучше
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer="adam", metrics=['accuracy'])

#model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))
accuracy = model.evaluate(x_val, y_val, verbose=0)
print('Test accuracy:', accuracy[1])

# Просмотр ошибок обучения. Оцениваем модель вручную.
for i in range(x_train.shape[0]):
    ans = model.predict(x_train[i].reshape((1,28,28,1))).reshape((10))
    ans = np.argmax(ans)
    label = np.argmax(y_train[i])
    if ans != label:
        plt.imshow(image.array_to_img(x_train[i]), cmap="gray") #
        plt.title("i:" + str(i) + "label:" + str(label) + "ans:" + str(ans))
        plt.show()

# Вобще для сверточных сетей в некоторых случаях можно делать размножение обучающей выборки путем вращения,
# отражения и др искажения изображений, иногда это даёт хороший рост качества

# Регуляризация - добавление в критерии качества к сумме квадратов ошибок сумму квадратов весов, сумму модулей
# весов с коэффициентом, и др. Добавление суммы квадратов к линейной или логистической регрессии называют
# ридж регрессией, добавление суммы модулей называется лассо, добавление и того и другого называют эластикнет
