# Набор данных состоит из изображений 37 пород домашних животных, по 200 изображений каждой породы
# (примерно по 100 изображений в тренировочном и тестовом сплитах). Каждое изображение включает соответствующие
# метки и попиксельные маски. Маски являются метками классов для каждого пикселя. Каждому пикселю присваивается
# одна из трех категорий:
# Класс 1: пиксель, принадлежащий питомцу.
# Класс 2: пиксель, граничащий с питомцем.
# Класс 3: ничего из вышеперечисленного/окружающий пиксель.

# pip install git+https://github.com/tensorflow/examples.git

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt

# скачиваем датасет
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


# нормализация к [0, 1], и заодно изменение меток пикселей с {1, 2, 3} на {0, 1, 2}
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


# изменение размера изображения к (128, 128)
def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


TRAIN_LENGTH = info.splits['train'].num_examples  # train и test уже разделены в датасете
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)


# слой аугментации
class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


# это называеися входной конвеер
train_batches = (
    train_images
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .repeat()
        .map(Augment())
        .prefetch(buffer_size=tf.data.AUTOTUNE))
test_batches = test_images.batch(BATCH_SIZE)


# визуализация
def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# смотрим на изображения и маски объектов
for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])

############################ СТРОИМ МОДЕЛЬ СЕГМЕНТАЦИИ
# загружаем обученную модель MobileNetV2, без dense слоев
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# берем из нее только первые несколько слоев
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',  # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# входной слой=входной слой, а выходные слои ВСЕ?
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

# это будут слои кодировщика, для них отключаем обучение
down_stack.trainable = False

# а это слои декодера из pix2pix
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]


# здесь построение модели энкодер-декодер
def unet_model(output_channels: int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])  # входной слой
    tf.keras.utils.plot_model(down_stack, show_shapes=True, to_file='down_stack.png')
    skips = down_stack(inputs)  # слои кодировщика
    x = skips[-1]
    skips = reversed(skips[:-1])  # reversed, кроме последнего слоя
    # Повышение частоты дискретизации и установление пропускных соединений
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()  # здесь паралельное соединение двух слоев в один
        x = concat([x, skip])
    # последний слой модели
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  # 64x64 -> 128x128
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


#################################### ОБУЧЕНИЕ
OUTPUT_CLASSES = 3
model = unet_model(output_channels=OUTPUT_CLASSES)
# выходов должно быть 128x128. from_logits=True, значит ф-я активации не нужна, кажый выход - одно число(целое????)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# !!!!!!!!!!!!! conda install graphviz pydot
tf.keras.utils.plot_model(model, show_shapes=True)  # рисуем модель (сохранит график в папку)

# пробуем предсказываь до обучения
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])
show_predictions()

# обратынй вызов для наблюдения прогресса обучения
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))

# обучение
EPOCHS = 2  # эпох можно взять больше, чем 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS
model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches,
                          callbacks=[DisplayCallback()])

# смотрим графики обучения
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

# смотрим на предсказания на тестовом примере
show_predictions(test_batches, 3)

# если классы взвесьть, то можно улучшить сегментацию
# https://www.tensorflow.org/tutorials/images/segmentation
