# Метод устаревающий, но может служить ориентиром, т.к. он математически понятный. Подходит в
# случаях, когда признаков небольшое количество и среди них нет малоинформативных
# Признаки обязательно нужно нормализовать
# В случае очень большой выборки требует много памяти для её хранения, зато к выборке несложно
# добаваить новые наблюдения
# Для заполнения пропусков sklearn.impute.KNNImputer
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold  # Генератор разбиений выборки
from sklearn.model_selection import cross_val_score  # Кросс-валидация
from sklearn.preprocessing import scale
import numpy as np

# Расширение файла не csv, но читает
# В этих данных нет названий столбцов, нужно об этом сказать конструктору
datawine = pd.read_csv("wine.data", header=None)
crossvalidator = KFold(n_splits=5, shuffle=True, random_state=42)  # Кросс-валидатор, делим выборку на 5 частей

x = datawine.drop(columns=0, axis=1)  # Выбрасываем столбец 1 с классами
x = scale(x)  # Нормировка значений. Очень улучшает обучение
y = datawine.filter(items=[0])  # Выбрасываем все столбцы, кроме 1

accuracylist = [0]
for neighbors in range(1, 50):
    kf = crossvalidator.split(x)  # Размеры разбиений [141 36]  [141 36]  [142 35]  [142 35]  [142 35]
    # leaf_size= параметр для выборки большей чем оперативная память
    # metric="mikowski" метрика рассстояния(Евклидово или др)
    # p=2 + metric="mikowski" - степень 2, значит Евклидово расстояние
    # weights="uniform" если выборка сбалансирована/несбалансирована
    neigh = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=4)  # n_jobs - количество ядер процессора
    results = cross_val_score(neigh, x, np.ravel(y.values.tolist()), cv=kf)
    accuracylist.append(sum(results) / len(results))

# confusion_matrix - матрица ошибок классификации
# classification_report - показывает precision, recall, f1-score
print(accuracylist)
print((max(accuracylist)))
print(accuracylist.index(max(accuracylist)))

# 1 0.73 - результат без нормировки
# 29 0.98 - с нормировкой
