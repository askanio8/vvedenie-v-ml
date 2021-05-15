# Использование tf.image
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

# Показываем 1-е изображение
get_label_name = metadata.features['label'].int2str
image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
plt.show()

# Просто функция для показа двух изображений
def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)
  plt.show()

# Отражение
flipped = tf.image.flip_left_right(image)
visualize(image, flipped)

# Просто image в оттенки серого
grayscaled = tf.image.rgb_to_grayscale(image)
# tf.squeeze просто удаляет размерности 1, если есть такие. Здесь это можно не ипользовать
visualize(image, tf.squeeze(grayscaled))

# Насыщенность
saturated = tf.image.adjust_saturation(image, 3)
visualize(image, saturated)

# Яркость
bright = tf.image.adjust_brightness(image, 0.4)
visualize(image, bright)

# Обрезка по центру
cropped = tf.image.central_crop(image, central_fraction=0.5)
visualize(image, cropped)

# Поворот
rotated = tf.image.rot90(image)
visualize(image, rotated)

# Выше были преобразованя изображений с заданным параметром изменения. Ниже - случайные преобразования
#tf.image.stateless_random_brightness
#tf.image.stateless_random_contrast
#tf.image.stateless_random_crop
#tf.image.stateless_random_flip_left_right
#tf.image.stateless_random_flip_up_down
#tf.image.stateless_random_hue
#tf.image.stateless_random_jpeg_quality
#tf.image.stateless_random_saturation

# Случайное преобразование яркости
for i in range(3):
  seed = (i, 0)  # tuple of size (2,)  # Почему-то seed - кортеж. Здесь результаты воспроизводятся!
  stateless_random_brightness = tf.image.stateless_random_brightness(
      image, max_delta=0.95, seed=seed)
  visualize(image, stateless_random_brightness)

# Случайное преобразование контраста
for i in range(3):
  seed = (i, 0)  # tuple of size (2,)
  stateless_random_contrast = tf.image.stateless_random_contrast(
      image, lower=0.1, upper=0.9, seed=seed)
  visualize(image, stateless_random_contrast)

# Случайная обрезка
for i in range(3):
  seed = (i, 0)  # tuple of size (2,)
  stateless_random_crop = tf.image.stateless_random_crop(
      image, size=[210, 300, 3], seed=seed)
  visualize(image, stateless_random_crop)

#################ПРИМЕР ПРЕОБРАЗОВАНИЯ ДАТАСЕТА##############################################
# Здесь получаем еще один, преобразованный датасет, которым можно дообучить сеть

# Определяем служебную функцию, которая приводит изображения к требуемому размеру для сети
IMG_SIZE = 180
def resize_and_rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = (image / 255.0)
  return image, label

# Пример функции случайных преобразований
def augment(image_label, seed):
  image, label = image_label
  image, label = resize_and_rescale(image, label)
  image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
  # Make a new seed
  new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
  # Random crop back to the original size
  image = tf.image.stateless_random_crop(
      image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
  # Random brightness
  image = tf.image.stateless_random_brightness(
      image, max_delta=0.5, seed=new_seed)
  image = tf.clip_by_value(image, 0, 1)
  return image, label

#
# Create counter and zip together with train dataset
# Дальше 2 непонятные строчки. Вобще здесь должно как-то задаваться зерно случайности для augmentation
# Хотя код работает. Поиграть с zip и с Counter
counter = tf.data.experimental.Counter()
train_ds = tf.data.Dataset.zip((train_ds, (counter, counter)))

batch_size = 64
train_ds = (
    train_ds
    .shuffle(1000)
    .map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
val_ds = (
    val_ds
    .map(resize_and_rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
test_ds = (
    test_ds
    .map(resize_and_rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)