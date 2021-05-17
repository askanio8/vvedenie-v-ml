import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import random

# Перемещаем картинки в папки своих классов
#cats = [shutil.move("./dogscats/train/train" + "/" + f, "./dogscats/train/cat" + "/" + f)
#       for f in os.listdir("./dogscats/train/train") if f.find("cat") != -1]
#dogs = [shutil.move("./dogscats/train/train" + "/" + f, "./dogscats/train/dog" + "/" + f)
#       for f in os.listdir("./dogscats/train/train") if f.find("dog") != -1]

# Выделяем из обучающей выборки выборку валидации
#cats = [f for f in os.listdir("./dogscats/train/cat")]
#dogs = [f for f in os.listdir("./dogscats/train/dog")]

#i = 0
#while i < 1500:
#    ind = random.randint(0, 12499)
#    if os.path.isfile("./dogscats/train/cat" + "/" + cats[ind]):
#        shutil.move("./dogscats/train/cat" + "/" + cats[ind], "./dogscats/validation/cat" + "/" + cats[ind])
#        i = i + 1

#i = 0
#while i < 1500:
#    ind = random.randint(0, 12499)
#    if os.path.isfile("./dogscats/train/dog" + "/" + dogs[ind]):
#        shutil.move("./dogscats/train/dog" + "/" + dogs[ind], "./dogscats/validation/dog" + "/" + dogs[ind])
#        i = i + 1


# Из папок делаем выборки обучающую и валидации. Параметры кроме первого - случайные преобразования.
# Для метода fit каждую эпоху будет генерироваться выборка со случайными преобразованиями в заданных пределах
train_datagen = ImageDataGenerator(rescale=1. / 255)


val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory("./dogscats/train", classes=['dog1000', 'cat1000'],
                                                    target_size=(150, 150), batch_size=20,
                                                    class_mode='binary', color_mode='grayscale')
val_generator = val_datagen.flow_from_directory("./dogscats/validation", classes=['dog1000', 'cat1000'],
                                                    target_size=(150, 150),
                                                    class_mode='binary', color_mode='grayscale')

# Модель
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu',
                  input_shape=(150,150,1)),
    layers.MaxPooling2D((3, 3)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((3,3)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((3, 3)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(1024, 'relu'),
    layers.Dense(1024, 'relu'),
    layers.Dense(1, 'sigmoid') # 1 нейрон – кошка или собака
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    train_generator,
    steps_per_epoch=50,
    epochs=30,
    validation_data=val_generator,
    callbacks=[tensorboard_callback])

# Дообучение на отфильтрованных данных