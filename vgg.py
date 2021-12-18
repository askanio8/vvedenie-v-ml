# argmax keys https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image

model = keras.applications.VGG16(include_top=True,  # использовать полносвязную часть сети в конце
                                 weights='imagenet',  # взять предобученные веса
                                 input_tensor=None,
                                 pooling=None,
                                 classes=1000,
                                 classifier_activation='softmax')


img = Image.open('img.jpg')
plt.imshow(img)
plt.show()

# приводим к входному формату VGG-сети
img = np.array(img)
x = keras.applications.vgg16.preprocess_input(img)
print(x.shape)
x = np.expand_dims(x, axis=0)

# прогоняем через сеть
res = model.predict(x)
print(np.argmax(res))