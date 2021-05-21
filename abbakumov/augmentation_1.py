import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Загружаем датасет
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# Количество классов
num_classes = metadata.features['label'].num_classes
print(num_classes)

# Выводим перый экземпляр
get_label_name = metadata.features['label'].int2str
image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
plt.show()

# Изменяем размер изображения на 180х180 и изменяем формат пикселей в диапазон [0,1]
IMG_SIZE = 180
# Здесь имеется ввиду создание сети из двух слоёв предобработки, её можно добавить в другую сеть обучения
# или использовать для предобработки
resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255)  # Пиксели в диапазон [0,1]
])
result = resize_and_rescale(image)
_ = plt.imshow(result)
plt.show()

# Проверяем, что пиксели находятся в [0-1] .
print("Min and max pixel values:", result.numpy().min(), result.numpy().max())
# Посмотреть значения
a = result.numpy()

# Ещё вариант сети предобработки с другими слоями(алгоритмами)
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),  # Отражение
  layers.experimental.preprocessing.RandomRotation(0.2),  # Вращение
])

# Add the image to a batch
# Добавляет в тензор ещё одно измерение в начало з
# Созданная сеть предобработки принимает тензор - коллекцию тензоров экземпляров, то есть тензор
# с размерностью на 1 больше. Здесь для эксперимента, мы работаем с одним экземпляром как с коллекцией
image = tf.expand_dims(image, 0)
plt.figure(figsize=(10, 1))  # 10 - это размер окна с картинками в дюймах
for i in range(9):
  augmented_image = data_augmentation(image)  # Выполняем случайные преобразования
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0])  # Выбираем первую картинку в коллекции с единственной картинкой
  plt.axis("off")  # Убираем шкалу с графиков
plt.show()
# Существует множество уровней предварительной обработки, которые вы можете использовать для увеличения данных,
# включая layers.RandomContrast , layers.RandomCrop , layers.RandomZoom и другие.
# Можно создать собственный слой layers.Lambda или унаследовать класс layers.Layer

# Пример для встраивания слоёв предобработки в сеть обучения
# Слои предобработки будут срабатывать только на этапе обучения и будут пропущены во вребя предсказания и оценки
# Эти слои сохраняются в модели при model.save
model = tf.keras.Sequential([
  resize_and_rescale,
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # Rest of your model
])

# А можно так - это просто предобработка - генерация ещё одной обучающей выборки с помощью функции map
aug_ds = train_ds.map(
  lambda x, y: (resize_and_rescale(x, training=True), y))
