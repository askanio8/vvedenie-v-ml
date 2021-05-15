import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

# Перемещаем картинки в папки своих классов
# cats = [shutil.move("./dogscats/train/train" + "/" + f, "./dogscats/train/cat" + "/" + f)
#        for f in os.listdir("./dogscats/train/train") if f.find("cat") != -1]
# dogs = [shutil.move("./dogscats/train/train" + "/" + f, "./dogscats/train/dog" + "/" + f)
#        for f in os.listdir("./dogscats/train/train") if f.find("dog") != -1]

# Из паок делаем выборки обучаюзую и валидации
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.1)
train_generator = datagen.flow_from_directory("./dogscats/train", classes=['dog', 'cat'],
                                                    target_size=(150, 150), batch_size=20,
                                                    class_mode='binary', subset='training')
val_generator = datagen.flow_from_directory("./dogscats/train", classes=['dog', 'cat'],
                                                    target_size=(150, 150), batch_size=20,
                                                    class_mode='binary', subset='validation')

# Модель
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu',
                  input_shape=(150,150,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(512, 'relu'),
    layers.Dense(1, 'sigmoid') # 1 нейрон – кошка или собака
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=val_generator,
    callbacks=[tensorboard_callback])
