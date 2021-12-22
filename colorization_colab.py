import numpy
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob

size = [256, 256, 3]
# переводим изображение в формат lab
def processed_image(img):
    image = img.resize((256, 256), Image.BILINEAR)
    image = np.array(image, dtype=float)
    size = image.shape
    lab = rgb2lab(1.0 / 255 * image)
    X, Y = lab[:, :, 0], lab[:, :, 1:]

    Y /= 128  # нормируем выходные значение в диапазон от -1 до 1
    X = X.reshape(size[0], size[1], 1)
    Y = Y.reshape(size[0], size[1], 2)
    return X, Y, size

model = load_model('F:/clrmodel')


# загружаем тестовые изображение
x = [Image.open(f) for f in glob.glob("F:/test/*.jpg")]  # открываем тестовые изображения
X = [z[0] for z in list(map(processed_image, x))]  # извлекаем их серые каналы
X = numpy.array(X)
X = numpy.expand_dims(X, axis=1)

outputs = [model.predict(i) for i in X]  # получаем каналы цвета
outputs = numpy.array(outputs)


outputs = [i * 128 for i in outputs]
min_vals, max_vals = -128, 127
ab = [np.clip(z[0], min_vals, max_vals) for z in outputs]

for x, X, ab in zip(x, X, ab):
    cur = np.zeros((size[0], size[1], 3))
    cur[:, :, 0] = np.clip(X[0][:, :, 0], 0, 100)
    cur[:, :, 1:] = ab
    plt.subplot(1, 2, 1)
    plt.imshow(x)
    plt.subplot(1, 2, 2)
    cur = lab2rgb(cur)
    x = numpy.array(x)
    plt.imshow(cur)
    plt.show()
