import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

import datetime

#%load_ext tensorboard


####################### иСПОЛЬЗОВАНИЕ ГОТОВОЙ СЕТИ
mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
#inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"

classifier_model = mobilenet_v2

IMAGE_SHAPE = (224, 224)

# оборачиваем готовую модель в слой, потом делаем его моделью
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])

# скачиваем картинку для примера
grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper.show()
grace_hopper = np.array(grace_hopper)/255.0
print(grace_hopper.shape)  # (224, 224, 3)

# добавляем еще одно измерение батча и подаем на распознавание
result = classifier.predict(grace_hopper[np.newaxis, ...])
print(result.shape)  # (1, 1001)
predicted_class = tf.math.argmax(result[0], axis=-1)
print(predicted_class)  # <tf.Tensor: shape=(), dtype=int64, numpy=653>

# расшифровуем предсказание по меткам набора данных ImageNet
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]  # класс Military uniform
_ = plt.title("Prediction: " + predicted_class_name.title())
plt.show()

####################################### ДАТАСЕТ ДЛЯ ДООБУЧЕНИЯ
# загружаем цветочный датасет
data_root = tf.keras.utils.get_file(
  'flower_photos',
  'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# метки классов
class_names = np.array(train_ds.class_names)
print(class_names)  # ['daisy' 'dandelion' 'roses' 'sunflowers' 'tulips']

# нормализация к диапазону [0; 1]  (стандарт TensorFlow Hub)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# буферизация и предобработка
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# смотрим на размерности одного батча
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)  # (32, 224, 224, 3)
  print(labels_batch.shape)  # (32,)
  break


# пробуем распознать тренировочный датасет. один из пяти классов цветов есть и в датасете, на котором была обучена модель
result_batch = classifier.predict(train_ds)
predicted_class_names = imagenet_labels[tf.math.argmax(result_batch, axis=-1)]
print(predicted_class_names)
# результаты ожидаемо плохие
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
image_batch, _ = next(iter(train_ds))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")
plt.show()

################################### ДООБУЧЕНИЕ
# скачиваем сверточные слои обученной модели
mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
#inception_v3 = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
feature_extractor_model = mobilenet_v2

feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model,
    input_shape=(224, 224, 3),
    trainable=False)

# смотрим на форму выходов последнего слоя
feature_batch = feature_extractor_layer(image_batch)  # подсвечивает но работает
print(feature_batch.shape)  # (32, 1280)

# добавляем к сверточным слоям слой классификации и оборачиваем все в модель
num_classes = len(class_names)
model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(num_classes)
])

# смотрим на форму выходов всей модели
predictions = model(image_batch)
print(predictions.shape)  # [32, 5]

# компилируем модель
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

# добавляем вызов tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1) # Enable histogram computation for every epoch.

# обучение. лучше взять эпох больше
NUM_EPOCHS = 10
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=NUM_EPOCHS,
                    callbacks=tensorboard_callback)
# можно запустить %tensorboard --logdir logs/fit

# пробуем классифицировать
predicted_batch = model.predict(image_batch)
predicted_id = tf.math.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
print(predicted_label_batch)

# смотрим и проверяем
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_label_batch[n].title())
  plt.axis('off')
_ = plt.suptitle("Model predictions")
plt.show()
# точность окло 90%. можно увеличить датасет, добавить аугментацию, тренировать последние сверточные слои

############################# СОХРАНЕНИЕ И ЗАГРУЗКА МОДЕЛИ
# сохранение модели
t = time.time()
export_path = "/tmp/saved_models/{}".format(int(t))
model.save(export_path)
print(export_path)

# загрузка модели
reloaded = tf.keras.models.load_model(export_path)

# проверка
result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)
abs(reloaded_result_batch - result_batch).max()
