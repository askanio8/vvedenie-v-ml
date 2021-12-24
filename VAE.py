import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import keras.backend as K
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Lambda, BatchNormalization, Dropout


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test  = np.reshape(x_test,  (len(x_test),  28, 28, 1))


hidden_dim = 2
batch_size = 60 # должно быть кратно 60 000

# два слоя для страховки от переобучения. Часто используются в VAE, не всегда полезны
def dropout_and_batch(x):
  return Dropout(0.3)(BatchNormalization()(x))

# энкодер
input_img = Input((28, 28, 1))
x = Flatten()(input_img)
x = Dense(256, activation='relu')(x)
x = dropout_and_batch(x)
x = Dense(128, activation='relu')(x)
x = dropout_and_batch(x)
z_mean = Dense(hidden_dim)(x)  # обучим получать на выходе этого слоя мат ожидание
z_log_var = Dense(hidden_dim)(x)  # обучим получать на выходе этого слоя логарифм дисперсии

# ф-я для лямбда слоя.
def noiser(args):
  global z_mean, z_log_var
  z_mean, z_log_var = args  # выходы двух параллельных предыдущих слоев z_mean, z_log_var
  # при умножении выходов одного из слоев на случайный вектор с нормальным распределением и сложении результата с вторым вектором
  # в результате обучения первый вектор будет стремиться к дисперсии а второй к мат ожиданию
  N = K.random_normal(shape=(batch_size, hidden_dim), mean=0., stddev=1.0)  # генерируем случайное отклонение для дисперсии
  return K.exp(z_log_var / 2) * N + z_mean  # exp(z_log_var / 2) это var просто дисперсия

h = Lambda(noiser, output_shape=(hidden_dim,))([z_mean, z_log_var])  # лямбда слой

# декодер
input_dec = Input(shape=(hidden_dim,))
d = Dense(128, activation='relu')(input_dec)
d = dropout_and_batch(d)
d = Dense(256, activation='relu')(d)
d = dropout_and_batch(d)
d = Dense(28*28, activation='sigmoid')(d)
decoded = Reshape((28, 28, 1))(d)

encoder = keras.Model(input_img, h, name='encoder')
decoder = keras.Model(input_dec, decoded, name='decoder')
vae = keras.Model(input_img, decoder(encoder(input_img)), name="vae")


# ф-я потерь
# ??по логике формулы kl_loss должен устремлять к 0 z_log_var и z_mean. А loss не даст это сделать полностью, в итоге найдется баланс
# математические ожидания и дисперсии будут уменьшены до минимальных значений, при которых декодер сможет различать класс изображения
def vae_loss(x, y):
  x = K.reshape(x, shape=(batch_size, 28*28))
  y = K.reshape(y, shape=(batch_size, 28*28))
  # первая компонента потерь loss - сумма квадратов разности между пикселями по каждому изображению. loss.shape = (batch_size,)
  loss = K.sum(K.square(x-y), axis=-1)
  # вторая компонента - дивиргенция Кульбака-Лейблера. Расстояние между информационными энтропиями матриц
  # в формуле операции над матрицами batch_size X hidden_dim
  kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
  return loss + kl_loss

vae.compile(optimizer='adam', loss=vae_loss)

vae.fit(x_train, x_train, epochs=5, batch_size=batch_size, shuffle=True)

# посмотрим на распределение выходов энкодера
h = encoder.predict(x_test[:6000], batch_size=batch_size)
plt.rcParams['figure.figsize'] = [20, 10]  # побольше окно графика
colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:purple',
          5:'tab:brown', 6:'tab:pink', 7:'tab:gray', 8:'tab:olive', 9:'tab:cyan'}

for i, c in colors.items():
  plt.scatter(h[y_test[:6000] == i, 0], h[y_test[:6000] == i, 1], s=10.0, c=c)

#plt.scatter(h[:, 0], h[:, 1])

# сгенерируем изображения по сетке дисперсий и мат ожиданий
n = 10
total = 2*n+1
plt.figure(figsize=(total, total))

num = 1
for i in range(-n, n+1):
  for j in range(-n, n+1):
    ax = plt.subplot(total, total, num)
    num += 1
    img = decoder.predict(np.expand_dims([3*i/n, 3*j/n], axis=0))  # берем разные мат ожидания и дисперсии и смотрим
    plt.imshow(img.squeeze(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)