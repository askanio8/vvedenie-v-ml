# Это как-то работает
import tensorflow as tf
import tensorflow_datasets as tfds

# здесь мы получаем данные в специальном формате, имеющем встроенные функции для оптимизации, а вот изучать их
# не очень удобно.
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,  # Здесь все в одном файле, но если файлов несколько, рекомендуют их так перемешивать
    as_supervised=True,  # возвращает кортеж (img, label) вместо dict {'image': img, 'label': label}
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # Данные из int в float
ds_train = ds_train.cache()  # Кэширует часть данных обучения, которая влазит в оперативную память
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)  # Перемешивает то что в кэше
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)  # prefetch устанавливает предзагрузку следующей
# части данных в кэш до окончания обучения на предыдущей части. AUTOTUNE параметр сам решает сколько и когда
# загружать данных

# что-то похожее делаем для тестовых данных
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

test_loss, test_acc = model.evaluate(ds_test)
print("Loss:", test_loss)
print("Accuracy:", test_acc)  # 97.5%
