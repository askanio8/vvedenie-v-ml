import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras

img = Image.open('img.jpg')
img_style = Image.open('img_style.jpg')

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(img_style)
plt.show()

# предобработка изображений под vgg (добавление размерности, перевод в BGR и др)
x_img = keras.applications.vgg19.preprocess_input(np.expand_dims(img, axis=0))
x_style = keras.applications.vgg19.preprocess_input(np.expand_dims(img_style, axis=0))

# функция, обратная к vgg19.preprocess_input (из BGR в RGB и др)
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)  # убираем лишнюю размерность

    # проверяем, что теперь марица изображения имеет три размерности,
    # если это не так то останавливаем работу программы
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    # лишнее дублирование??
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # обратная предобработка с добавлением средних значений по каждому каналу
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # так можно перевести из BGR в RGB (вобщето есть спецфункции для этого...)
    x = x[:, :, ::-1]

    # если есть значения выходящие за пределы диапазона 0-255, то устанавливаем их в 0 и 255 соответственно
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# берем саму сеть уже натренированную без последних слоев Dense
vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Имя слоя, по которому будем сравнивать соответствие содержимого
content_layers = ['block5_conv2']

# Имена слоев, по которым будем сравнивать соответствие стиля
# имена нужных слоев можно узнать в summary
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# теперь получаем ссылки на сами слои из vgg по их именам
style_outputs = [vgg.get_layer(name).output for name in style_layers]
content_outputs = [vgg.get_layer(name).output for name in content_layers]
model_outputs = style_outputs + content_outputs

print(vgg.input)
for m in model_outputs:
    print(m)
# делаем такую же сеть из vgg как она сама, только выходными будут все указанные слои
model = keras.models.Model(vgg.input, model_outputs)
for layer in model.layers:
    layer.trainable = False

print(model.summary())  # вывод структуры НС в консоль

# получаем признаки контента для изображения контента и признаки стиля для изображения стиля
def get_feature_representations(model):
    # batch compute content and style features
    style_outputs = model(x_style)
    content_outputs = model(x_img)

    # Get the style and content feature representations from our model
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features

# вычисляет потери по контенту
# base_content - выходы последнего слоя по исходному изображению
# target - выходы последнего слоя по преобразованному изображению
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))  # среднее значение квадрата разности тензоров


# матрица Грама. каждый элемент характеризует схожесть всех карт признаков друг с другом
# input_tensor - набор карт признаков
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])  # берем последнюю размерность(количество карт признаков)
    # остальные размерности разворачиваем в вектор и получаем одну матрицу
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]  # берем первую размерность(длину вектора)
    gram = tf.matmul(a, a, transpose_a=True) # перемножаем матрицу саму на себя. рез-тат n x n
    return gram / tf.cast(n, tf.float32)  # делим каждый элемент матрицы на n

# вычисляем функциию потерь для каждого выходного слоя - квадрат разности между матрицами Грама
# преобразуемого изображения и стилевого
def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))


# общая функция потерь
# loss_weights - кортеж из двух параметров a и b (соотношение требуемого стиля и контента)
# init_image - формируемое изображение
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights  # распаковываем a и b

    model_outputs = model(init_image)  # пропускаем преобразованоое? изображение через сеть

    style_output_features = model_outputs[:num_style_layers]  # получаем признаки стиля
    content_output_features = model_outputs[num_style_layers:]  # получаем признаки контента

    style_score = 0  # потери стиля
    content_score = 0  # потери контента

    # потерям для каждого слоя стиля даем одинаковый вес(1/5.0 = 0.2)
    weight_per_style_layer = 1.0 / float(num_style_layers)
    # вычисляем потери для каждого слоя стиля, взвешиваем и складываем
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # вычисляем потери для каждого слоя контента, взвешиваем и складываем(хотя тут цикл не нужен т.к. слой один)
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    # взвешиваем относительные потери стиля и контента
    style_score *= style_weight
    content_score *= content_weight

    # суммируем потери
    loss = style_score + content_score
    return loss, style_score, content_score


num_iterations = 100
content_weight = 1e3
style_weight = 1e-2

# получаем признаки контента для изображения контента и признаки стиля для изображения стиля
style_features, content_features = get_feature_representations(model)
# считаем матрицы Грама для признаков стиля
gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

# берем изображение, которое будем преобразовывать
init_image = np.copy(x_img)
# переводим его в формат tensorflow
init_image = tf.Variable(init_image, dtype=tf.float32)

# выбираем опримизатор
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=2, beta1=0.99, epsilon=1e-1)
iter_count = 1  # счетчик итераций
best_loss, best_img = float('inf'), None  # начальные значения лучших потерь и лучшего изображения
loss_weights = (style_weight, content_weight)  # упаковываем веса

cfg = {  # параметры в словарь для удобства
    'model': model,
    'loss_weights': loss_weights,
    'init_image': init_image,
    'gram_style_features': gram_style_features,
    'content_features': content_features
}

# параметры для преобразования из BGR в RGB
norm_means = np.array([103.939, 116.779, 123.68])
min_vals = -norm_means
max_vals = 255 - norm_means

imgs = []  # список всех созданных изображений

# цикл градиентного спуска - формирование изображений
for i in range(num_iterations):
    with tf.GradientTape() as tape: # запоминаем градиенты относительно пикселей изображения(тут пиксели это параметры)
        all_loss = compute_loss(**cfg)  # в процессе вычисления потерь

    loss, style_score, content_score = all_loss
    grads = tape.gradient(loss, init_image)  # вычисление градиентов

    opt.apply_gradients([(grads, init_image)])  # применяем градиент к пикселям изображения
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)  # ограничиваем пиксели диапазоном 0-255
    init_image.assign(clipped)  # новое изображение в переменную init_image

    # запоминаем лучшие потери и лучшее изображение
    if loss < best_loss:
        # Update best loss and best image from total loss.
        best_loss = loss
        best_img = deprocess_img(init_image.numpy())

        # Use the .numpy() method to get the concrete numpy array
        plot_img = deprocess_img(init_image.numpy())
        imgs.append(plot_img)
        print('Iteration: {}'.format(i))

plt.imshow(best_img)
plt.show()
print(best_loss)

#image = Image.fromarray(best_img.astype('uint8'), 'RGB')
#image.save("result.jpg")