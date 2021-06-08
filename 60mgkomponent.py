# Кроме метода главных компонент существует еще собственно факторный анализ, многомерное шкалирование
# sklearn.manifold.MDS, и др. Используются для понижения размерности, проекции многомерного облака точек в
# пространство меньшей размерности с сорхранением расстояний,
# измерения неизмеримого(любовь, лояльность, характер...)...
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np


closeData = pd.read_csv('close_prices.csv', index_col='date')
# Из 30 признаков делаем 10
pca = PCA(n_components=10)
pca.fit(closeData)
# Процент дисперсии, который объясняет каждая компонента
print(pca.explained_variance_ratio_)  # Первые 4 компоненты отвечают за больше чем 90% дисперсии
# Процент вкалада каждого исходного признака в каждую результирующую компоненту
print(pca.components_)
# Признак с самым большим вкладом в первую компоненту
print(np.argmax(pca.components_[0]))
# Преобразованные признаки, из 30 получилось 10
transformedCloseData = pca.transform(closeData)

djData = pd.read_csv('djia_index.csv', index_col='date')
# Корреляция Пирсона между первой компонентой и индексом Доу Джонса
corrPirson = np.corrcoef(transformedCloseData[:, 0], np.ravel(djData.values))  # 0.91
print(corrPirson)