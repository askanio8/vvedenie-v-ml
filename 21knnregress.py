from sklearn import datasets
import numpy as np
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold  # Генератор разбиений выборки
from sklearn.model_selection import cross_val_score  # Кросс-валидация
import pandas as pd

bostondata = datasets.load_boston()
data = np.c_[bostondata.data, bostondata.target]  # Достаём данные
columns = np.append(bostondata.feature_names, ["target"])  # Достаём заголовки
bostondataframe = pd.DataFrame(data, columns=list(columns))  # Создает таблицу из данных и заголовков
target = bostondataframe.filter(items=["target"])  # Y
bostondataframe = bostondataframe.drop(columns="target")  # X
bostondataframe = scale(bostondataframe)  # Нормируем

crossvalidator = KFold(n_splits=5, shuffle=True, random_state=42)

accuracylist = {}
plist = np.linspace(1, 10, num=200)  # p - степень в метрике Минковского
for p in plist:
    kf = crossvalidator.split(bostondataframe)
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, n_jobs=4)  # Метрика Минковского по умолчанию
    # Среднеквадратичная ошибка scoring='neg_mean_squared_error'
    result = cross_val_score(neigh, bostondataframe, target, scoring='neg_mean_squared_error')
    accuracylist.update({p: sum(result) / len(result)})

# Лучшая степень для метрики Минковского
print(max(accuracylist, key=accuracylist.get))  # Индекс максимального значения в словаре
