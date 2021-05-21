import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import random
import numpy
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import datetime

# Перемещаем картинки в папки своих классов
# cats = [shutil.move("./dogscats/train/train" + "/" + f, "./dogscats/train/cat" + "/" + f)
#       for f in os.listdir("./dogscats/train/train") if f.find("cat") != -1]
# dogs = [shutil.move("./dogscats/train/train" + "/" + f, "./dogscats/train/dog" + "/" + f)
#       for f in os.listdir("./dogscats/train/train") if f.find("dog") != -1]

# Выделяем из обучающей выборки выборку валидации
# cats = [f for f in os.listdir("./dogscats/train/cat")]
# dogs = [f for f in os.listdir("./dogscats/train/dog")]

# i = 0
# while i < 1500:
#    ind = random.randint(0, 12499)
#    if os.path.isfile("./dogscats/train/cat" + "/" + cats[ind]):
#        shutil.move("./dogscats/train/cat" + "/" + cats[ind], "./dogscats/validation/cat" + "/" + cats[ind])
#        i = i + 1

# i = 0
# while i < 1500:
#    ind = random.randint(0, 12499)
#    if os.path.isfile("./dogscats/train/dog" + "/" + dogs[ind]):
#        shutil.move("./dogscats/train/dog" + "/" + dogs[ind], "./dogscats/validation/dog" + "/" + dogs[ind])
#        i = i + 1

# Удаляем поврежденные изображаения или не jpeg формата внутри
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Эта строка решает ошибку trunkated images, если таких файлов немного,
# если много, то лучше искть другое решение, т.к результаы обучения могут быть не очень
#boys = [f for f in os.listdir("./dataset/train/boys")]
#girls = [f for f in os.listdir("./dataset/train/girls")]
import imghdr
#for i, file in enumerate(boys):
#    a = imghdr.what(os.path.join("./dataset/validation/boys", file))
#    if a != 'jpeg':
#        os.remove(os.path.join("./dataset/validation/boys", file))
#for i, file in enumerate(girls):
#    a = imghdr.what(os.path.join("./dataset/validation/girls", file))
#    if a != 'jpeg':
#        os.remove(os.path.join("./dataset/validation/girls", file))

# Проверяем размеры
#for i, file in enumerate(girls):
#    im = Image.open(os.path.join("./dataset/train/girls", file))
#    width, height = im.size
#    if width < 240 or height <240:
#        raise ValueError
#for i, file in enumerate(boys):
#    im = Image.open(os.path.join("./dataset/train/boys", file))
#    width, height = im.size
#    if width < 240 or height <240:
#        raise ValueError



BATCH_SIZE = 220
# Из папок делаем выборки обучающую и валидации. Параметры кроме первого - случайные преобразования.
# Для метода fit каждую эпоху? будет генерироваться выборка со случайными преобразованиями в заданных пределах
train_datagen = ImageDataGenerator(rescale=1. / 255,  #
                                   #rotation_range=40,  # скорее хуже
                                   width_shift_range=.2, # скорее лучше
                                   height_shift_range=.2, # скорее лучше
                                   #shear_range=.2,  # скорее хуже, замедляет
                                   #zoom_range=.2,  # скорее хуже
                                   horizontal_flip=True,  # лучше
                                   )

adrtrain = "./dogscats/train"
train_generator = train_datagen.flow_from_directory(adrtrain, classes=['dog', 'cat'],
                                                    target_size=(150, 150), batch_size=BATCH_SIZE,
                                                    class_mode='binary', color_mode='grayscale') #
# Смотрим на обучающую выборку
for i in range(1):
    plt.imshow(image.array_to_img(train_generator[int(i / BATCH_SIZE)][0][i % BATCH_SIZE]), cmap="gray") #
    plt.title(" Адрес:" + str(train_generator.filenames[train_generator.index_array[i]]))
    plt.show()

# Выборка валидации
adrval = "./dogscats/validation"
val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(adrval, classes=['dog', 'cat'],
                                                target_size=(150, 150), batch_size=BATCH_SIZE,
                                                class_mode='binary', color_mode='grayscale') #

# Модель
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=(150, 150, 1)),  # (150, 150, 1) если color_mode='grayscale'
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((3, 3)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((4, 4)),
    layers.Flatten(),
    layers.Dense(512, 'relu'),
    layers.Dense(1, 'sigmoid')  # 1 нейрон – кошка или собака
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + "Maxpool3x3lastlayer"#datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=15,
    validation_data=val_generator,
    callbacks=[tensorboard_callback])

# Показываем картинки подряд с результатами распознавания
for i in range(120):
    res = model.predict(val_generator[int(i / BATCH_SIZE)][0][i % BATCH_SIZE].reshape((1, 150, 150, 1)),
                        batch_size=1)
    plt.imshow(image.array_to_img(val_generator[int(i / BATCH_SIZE)][0][i % BATCH_SIZE]), cmap="gray")  #
    # 0 - собака, 1 - кошка
    plt.title("Вероятность:" + str(res[0][0]) +
              " Адрес:" + str(val_generator.filenames[val_generator.index_array[i]]))
    plt.show()

# Дообучение на отфильтрованных данных
# flow_from_dataframe

add_train_datagen = ImageDataGenerator(rescale=1. / 255)
add_train_generator = train_datagen.flow_from_directory("./dogscats/train", classes=['dog1000add', 'cat1000add'],
                                                        target_size=(150, 150), batch_size=BATCH_SIZE,
                                                        class_mode='binary', color_mode='grayscale')
