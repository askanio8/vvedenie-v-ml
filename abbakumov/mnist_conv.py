import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers


# Здесь нет спецформата, обычные ndarray
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Смотрим на элемент с индексом 0
plt.imshow(x_train[0, :, :])
plt.colorbar()
plt.show()


# Добавляем размерность для соответствия tensorflow. 1 - потому что одно число - оттенок серого.
# Если бы было RGB, то 3
x_train = x_train.reshape((60000, 28, 28, 1))
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test.astype('float32') / 255


# Метки классов преобразуем в 10 категориальных признаков
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Создаём модель CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', # (3,3) - фильтр, ядро
                        input_shape=(28,28,1)),
    # conv2d (Conv2D)              (None, 26, 26, 32)        320
    # из тензора размерностью (28,28,1) получаем тензор (26, 26, 32)
    # 26 - это (28 - половина ширины ядра)
    # 32 - выбранное количество ядер
    # Знчение весов Param=320, это 32 ядра * 9 размер ядра + 32 свободных членов
    layers.MaxPooling2D((2,2)), # фильтр (2,2) для пулинга # уменьшение размера в 4 раза
    # max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
    layers.Conv2D(64, (3,3), activation='relu'),  # второй слой свертки
    # conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496
    # весов 18496 - это ((3*3)*32+1)*64
    layers.MaxPooling2D((2,2)),  # уменьшение размера ещё в 4 раза
    # max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
    # почему именно (5, 5) хз вобще должно быть 11/2, но так не делится
    layers.Conv2D(64, (3,3), activation='relu'),  # третий слой свертки
    # conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928
    # весов 36928 - это ((3*3)*64+1)*64
    layers.Flatten(),  # Вытягиваем тензор в вектор пред подачей в обычные слои
    # flatten (Flatten)            (None, 576)               0
    # выход 576 - это 3*3*64
    layers.Dense(64, 'relu'),  # первый обычный слой
    # dense (Dense)                (None, 64)                36928
    layers.Dense(10, 'softmax')  # второй обычный слой
    # dense_1 (Dense)              (None, 10)                650
])

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss:", test_loss)
print("Accuracy:", test_acc)  # 99%
