import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.datasets import mnist
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[y_train == 7]  # берем только семерки
y_train = y_train[y_train == 7]

BUFFER_SIZE = x_train.shape[0]  # всего изображений
BATCH_SIZE = 100

BUFFER_SIZE = BUFFER_SIZE // BATCH_SIZE * BATCH_SIZE  # берем число изображений кратное 100
x_train = x_train[:BUFFER_SIZE]
y_train = y_train[:BUFFER_SIZE]
print(x_train.shape, y_train.shape)

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# перемешиваем датасет и разбиваем на батчи заранее
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# формирование сетей
hidden_dim = 2

# генератор
generator = tf.keras.Sequential([
    Dense(7 * 7 * 256, activation='relu', input_shape=(hidden_dim,)),
    BatchNormalization(),  # нормализация данных в скрытом слое, хорошо для генерации изображений
    Reshape((7, 7, 256)),  # изменение формы для след сверточного слоя
    # свертка 128 ядер размером 5x5 с шагом 1(как я понял в Transpose слое strides это сдвиг расположения признаков
    # в картах предыдущего слоя. (1, 1) - это плотно, (2, 2) - в шахматном порядке и тд). strides>1 приводит к
    # увеличению изображения. Тут 7х7, далее после strides=(2, 2) -> 14x14, на выходе сети уже 28x28
    # Transpose для обратного прохода
    Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', activation='relu'),  # выход 128х7х7
    BatchNormalization(),
    # свертка 64 ядра размером 5x5 с шагом 2
    Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),  # выход 64х14х14
    BatchNormalization(),
    # свертка 1 ядро размером 5x5 с шагом 2
    Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='sigmoid'),  # выход 1х28х28
])

# дискриминатор
discriminator = tf.keras.Sequential()
discriminator.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.3))

discriminator.add(Flatten())
discriminator.add(Dense(1))  # здесь линейная ф-я активации, вроде бы для лучшего обучения

# потери
# создаём экземпляр класса?
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    # tf.ones_like(fake_output) - создаём матрицу из единиц, по форме матрицы fake_output
    # fake_output - это вектор ответов дискриминатора при подаче ему на вход батча выходов генератора
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss


def discriminator_loss(real_output, fake_output):
    # real_output - вектор ответов дискриминатора при подаче ему на вход батча реальных изображений
    # fake_output - вектор ответов дискриминатора при подаче ему на вход батча выходов генератора
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)  # сравниваем с вектором единиц
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)  # сравниваем с вектором нулей
    total_loss = real_loss + fake_loss
    return total_loss


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# обучение
@tf.function
def train_step(images):  # ф-я шага обучения  # images - батч реальных изображений
    noise = tf.random.normal([BATCH_SIZE, hidden_dim])  # батч шума для входов генератора

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:  # считать градиенты
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # вычисляем градиенты, в параметры передаём ф-ю потерь и настраиваемые параметры
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # применяем градиенты
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss  # возвращаем потоери для контроля за скоростью обучения


def train(dataset, epochs):  # это вместо ф-ии fit
  history = []
  MAX_PRINT_LABEL = 10
  th = BUFFER_SIZE // (BATCH_SIZE * MAX_PRINT_LABEL)

  for epoch in range(1, epochs + 1):  # цикл обчучения по эпохам
    print(f'{epoch}/{EPOCHS}: ', end='')  # end = '' не переводит на новую строку

    start = time.time()
    n = 0

    gen_loss_epoch = 0
    for image_batch in dataset:
      gen_loss, disc_loss = train_step(image_batch)
      gen_loss_epoch += K.mean(gen_loss)
      if (n % th == 0): print('=', end='')  # показываем прогресс после каждого батча
      n += 1

    history += [gen_loss_epoch / n]
    print(': ' + str(history[-1]))
    print('Время эпохи {} составляет {} секунд'.format(epoch, time.time() - start))

  return history


# запуск процесса обучения
EPOCHS = 50
history = train(train_dataset, EPOCHS)

plt.plot(history)
plt.grid(True)
plt.show()

# отображение результатов генерации
n = 2
total = 2 * n + 1

plt.figure(figsize=(total, total))

num = 1
for i in range(-n, n + 1):
  for j in range(-n, n + 1):
    ax = plt.subplot(total, total, num)
    num += 1
    img = generator.predict(np.expand_dims([0.5 * i / n, 0.5 * j / n], axis=0))
    plt.imshow(img[0, :, :, 0], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
# в colab учится в 10 раз быстрее

# В итоге генератор генерирует изображения, взглядом вполне отличимые от реальных. Потери генератора и
# дискриминатора получаюбтся равными, можно попробовать дать больший вес потерям генератора
