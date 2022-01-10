import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np


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

# выкидываем выбросы
todrop = [28162, 46857, 39184, 38680, 12830, 23588, 28710, 16678, 34861, 12078, 36141, 10800, 40752, 30770,
          51508, 6714, 23868, 26940, 36934, 53063, 30792, 59720, 49487, 51280, 38230, 32342, 15450, 13662,
          46432, 30049, 48228, 43109, 34404, 26471, 2148, 23911, 57972, 2676, 15991, 1404, 26748, 20350,
          38526, 49026, 33412, 132, 35464, 21906, 1940, 15766, 58022, 35246, 58802, 54458, 52927, 37834,
          52938, 6092, 27085, 40654, 56014, 3532, 39378, 16376, 24798, 33506, 38370, 51944, 32747, 47340,
          494, 41718, 49143, 47094, 13558, 46078]
np.delete(x_train, todrop)
np.delete(y_train, todrop)


batch_size = 48
train_datagen = ImageDataGenerator(#rescale=1. / 255,  #
                                   rotation_range=10,  # скорее хуже
                                   width_shift_range=.09, # скорее лучше
                                   height_shift_range=.09, # скорее лучше
                                   #shear_range=.2,  # скорее хуже, замедляет
                                   #zoom_range=.2,  # скорее хуже
                                   #horizontal_flip=True,  # лучше
                                   )
train_datagen.fit(x_train)
train_datagen = train_datagen.flow(x_train,y_train,batch_size=batch_size)

val_datagen = ImageDataGenerator().flow(x_test,y_test,batch_size=batch_size)



# Создаём модель CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', # (3,3) - фильтр, ядро
                        input_shape=(28,28,1)),
    # conv2d (Conv2D)              (None, 26, 26, 32)        320
    # из тензора размерностью (28,28,1) получаем тензор (26, 26, 32)
    # 26 - это (28 - половина ширины ядра)
    # 32 - выбранное количество ядер
    # Знчение весов Param=320, это 32 ядра * 9 размер ядра + 32 свободных членов
    #layers.MaxPooling2D((2,2)), # фильтр (2,2) для пулинга # уменьшение размера в 4 раза
    # max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
    layers.Conv2D(64, (3,3), activation='relu'),  # второй слой свертки
    # conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496
    # весов 18496 - это ((3*3)*32+1)*64
    layers.MaxPooling2D((2,2)),  # уменьшение размера ещё в 4 раза
    # max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
    # почему именно (5, 5) хз вобще должно быть 11/2, но так не делится
    layers.Dropout(0.25),
    #layers.Conv2D(64, (3,3), activation='relu'),  # третий слой свертки
    # conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928
    # весов 36928 - это ((3*3)*64+1)*64
    layers.Flatten(),  # Вытягиваем тензор в вектор пред подачей в обычные слои
    # flatten (Flatten)            (None, 576)               0
    # выход 576 - это 3*3*64
    layers.Dense(512, 'relu'),  # первый обычный слой
    # dense (Dense)                (None, 64)                36928
    layers.Dropout(0.5),
    layers.Dense(10, 'softmax')  # второй обычный слой
    # dense_1 (Dense)              (None, 10)                650
])

print(model.summary())
Adam = keras.optimizers.Adam(
    learning_rate=0.001,  # по умолчанию лучше
    beta_1=0.85,  # скорость затухания 1-го момента(очень большое уменьшение замедляет и ухудшает)
    beta_2=0.999, # скорость затухания 2-го момента(уменьшение замедляет и ухудшает)
    epsilon=1e-07, # чем больше эпсилон, тем меньше скорость обучения вроде бы. По умолчанию лучше
    amsgrad=True,  # улучшенный adam - хорошо, быстро сходится, стабильно лучше, было 99.56
    name="Adam",)
# "Adagrad" по умолчанию - скрость обучения сильно падает возле 99.0, подбирал параметры - тоже не очень
Adagrad = keras.optimizers.Adagrad(
    learning_rate=0.001,
    initial_accumulator_value=0.1,
    epsilon=1e-07,
    name="Adagrad",)
# "Adadelta" по умолчанию - около 90 всего лишь, подбирал параметры - тоже не очень
Adadelta = keras.optimizers.Adadelta(learning_rate=0.001, # лучше не трогать, мало влияет вроде
                                     rho=0.99, # чем меньше, тем медленнее растет
                                     epsilon=1e-07,
                                     name="Adadelta")
#  Результат, как и обычного adam
Adamax = keras.optimizers.Adamax(
    learning_rate=0.001,
    beta_1=0.85,
    beta_2=0.999,
    epsilon=1e-07,
    name="Adamax",)
#  Результат, как и обычного adam, но показался медленнее
Nadam = keras.optimizers.Nadam(
    learning_rate=0.001,  # 0.0001- 99.31
    beta_1=0.85, # 99.50
    beta_2=0.999,
    epsilon=1e-07,
    name="Nadam",)
# "SGD" без параметров - неплохо 99.2
SGD = keras.optimizers.SGD(lr=0.001,  # так лучше
                                nesterov=True, # c нестеровым лучше
                                momentum=0.0) # плохо # чем меньше, тем меньше скорость почему-то
# "RMSprop" без параметров бысто учится но без рекородов по результату(98-99),
# подбирал параметры - тоже не очень
RMSprop = keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,  # нет разницы
    name="RMSprop",)
# "Ftrl" - По умолчанию совсем плохой результат, нужно найти рабочие параметры
Ftrl = keras.optimizers.Ftrl(
    learning_rate=0.001,
    learning_rate_power=-0.5,
    initial_accumulator_value=0.1,
    l1_regularization_strength=0.0,
    l2_regularization_strength=0.0,
    name="Ftrl",
    l2_shrinkage_regularization_strength=0.0,
    beta=0.0,)

model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Это для логирования в tensorboard
# Всё-таки нужно сохранять в разные папки каждое обучение. Или удалять созданные файлы перед каждым вызовом
log_dir = "logs/fit/" + 'Ftrl'  # + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_datagen, epochs=15,
          validation_data=val_datagen,
          callbacks=[tensorboard_callback])

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss:", test_loss)
print("Accuracy:", test_acc)  # 99.43% - 99.56%  # а если выкинуть выбросы, то 99.44% - 99.59%
# в терминале tensorboard --logdir=logs/fit/


# Дообучение. При любых условиях здесь становится хуже, скорее всего бесполезно
# sample_weightss.ipynb
answers = model.predict(x_train)
gen = [i for i, x in enumerate(answers) if x.max() < 0.9 and y_train[i].argmax() != x.argmax()]
new_x_train = x_train[gen]
new_y_train = y_train[gen]

model.fit(new_x_train, new_y_train, epochs=15,
          validation_data=val_datagen,
          callbacks=[tensorboard_callback])

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss:", test_loss)
print("Accuracy:", test_acc)


# Просмотр ошибок обучения. Оцениваем модель вручную.
for i in range(x_train.shape[0]):
    ans = model.predict(x_train[i].reshape((1,28,28,1))).reshape((10))
    ans = np.argmax(ans)
    label = np.argmax(y_train[i])
    if ans != label:
        plt.imshow(image.array_to_img(x_train[i]), cmap="gray") #
        plt.title("i:" + str(i) + "label:" + str(label) + "ans:" + str(ans))
        plt.show()
