import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test  = np.reshape(x_test,  (len(x_test),  28, 28, 1))

input_img = Input(shape=(28, 28, 1))
x = Flatten()(input_img)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
encoded = Dense(2, activation='linear')(x)  # кодируем в вектор размерности 2

input_enc = Input(shape=(2,))
d = Dense(64, activation='relu')(input_enc)
d = Dense(28*28, activation='sigmoid')(d)
decoded = Reshape((28, 28, 1))(d)

encoder = keras.Model(input_img, encoded, name="encoder")  # отдельно модель энкодер
decoder = keras.Model(input_enc, decoded, name="decoder")  # отдельно модель декодер
autoencoder = keras.Model(input_img, decoder(encoder(input_img)), name="autoencoder")  # так автоэнкодер
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# все три модели обучаются вместе
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=64,
                shuffle=True)

h = encoder.predict(x_test)
plt.rcParams['figure.figsize'] = [20, 10]  # побольше окно графика
colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:purple',
          5:'tab:brown', 6:'tab:pink', 7:'tab:gray', 8:'tab:olive', 9:'tab:cyan'}

for i, c in colors.items():
  plt.scatter(h[y_test == i, 0], h[y_test == i, 1], s=2.0, c=c)  # это я красиво сделал
plt.show()  # смотрим на график распределения закодированных векторов


img = decoder.predict(np.expand_dims([50, 250], axis=0))
plt.imshow(img.squeeze(), cmap='gray')
plt.show()