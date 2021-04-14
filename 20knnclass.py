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
    neigh = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=4)  # n_jobs - количество ядер процессора
    results = cross_val_score(neigh, x, np.ravel(y.values.tolist()), cv=kf)
    accuracylist.append(sum(results) / len(results))


print(accuracylist)
print((max(accuracylist)))
print(accuracylist.index(max(accuracylist)))

# 1 0.73 - результат без нормировки
# 29 0.98 - с нормировкой
