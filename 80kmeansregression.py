# Здесь метод кластеризации к-средних
# Есть ещё иерархический кластерный анализ и DBSCAN
# Для измерений расстояния в разных методах кластеризации используется Евклидово(разной степени) и
# расстояние Манхэттена. Последнее используется, когда для кластеризации более важно разница по
# комплексу параметров, а Евклидово - кода больше имеет значение величина отличия хотя бы по одному
# из параметров
# На выбор метода влияет также ожидаемая форма кластеров. Например шаровые скопления - метод Варда,
# средние невзвешенные расстояния и др. Ленточные скопления - метод ближайшего соседа
from skimage.io import imread
import skimage
import pylab
import numpy as np
from sklearn.cluster import KMeans


# Уменьшение цветов изображения
def convertImg(img, marks, centers):
    # Значениям RGB пикселей присваиваем RGB центров их кластеров
    # На самом деле между центрами кластеров и средними значениями есть очень маленькая разница почему-то, забил
    averageColorImage = np.asarray([centers[n] for n in marks])

    # Здесь находит медиану по R, G и B отдельно, возможно нужно искать именно медианный пиксель и брать его
    # значения? и упорядочивать тогда по евклидову расстоянию от 0?
    medianColorImage = img.copy()
    for i in range(len(centers)):
        medianColorImage[marks == i] = np.median(img[marks == i], axis=0)

    return medianColorImage, averageColorImage


# Вычисление метрики отношения пикового уровня сигнала к шуму
# вместо этого можно from skimage.metrics import peak_signal_noise_ratio
def PSNR(a, b):
    mse = np.mean((a - b) ** 2)
    psnr = 20 * np.log10(1 / mse ** 0.5)  # Преобразовать можно в psnr = -10 * np.log10(mse)
    return psnr


image = imread('parrots.jpg')  # Получаем массив [474, 713, 3]
image = skimage.img_as_float(image)  # Нормировка пикселей к диапазону [0, 1]
image = np.reshape(image, (-1, 3))  # Пиксели в колонку

cls = KMeans(random_state=241, n_clusters=11)  # Кластеризация
marks = cls.fit_predict(image)  # Почучаем столбец с метками кластеров

# Получаем изображения с медианными и средними цветами кластеров
# cluster_centers_ - значения центров кластеров
medianColorImage, averageColorImage = convertImg(image, marks, cls.cluster_centers_)

# Метрика PSNR в случае изображений измеряет разницу(например между сжатым и исходным изображением)
# обычные значения 30-60. Традиционно измеряется в децибелах Db
medianPsnr = PSNR(image, medianColorImage)
averagePsnr = PSNR(image, averageColorImage)  # Если кластеров не меньше 11, то PSNR > 10

# Рисуем
f = pylab.figure()
f.add_subplot(1, 2, 1)
pylab.imshow(averageColorImage.reshape((474, 713, 3)))
f.add_subplot(1, 2, 2)
pylab.imshow(medianColorImage.reshape((474, 713, 3)))
pylab.show()
print(medianPsnr, averagePsnr)