import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

data = pd.read_csv('gbm-data.csv')
dataArray = data.values  # Dataframe в nparray
Xdata = dataArray[:, 1:]
Ydata = dataArray[:, 0]
# Авторазбиение выборки на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size=0.8, random_state=241)

# Градиентыный бустинг это когда следующее дерево предсказывает ошибку предыдущего, в результате
# суммарная ошибка уменьшатся
# verbose=True выводит потери в консоль в процессе построения n_estimators=250 моделей
clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=0.2)
clf.fit(X_train, y_train)

# staged_decision_function - На каждой итерации обучения из 250 даём на распознавание выборку тестовую или
# обучающую, назад получаем ответы распознавания в формате оценок принадлежности классу а не список классов
# К полученным данным применяем функцию сигмоиды для нормировки в диапазон [1, 0], получаем вероятности
# принадлежности классам
sdfTrain = 1 / (1 + np.exp(-np.asarray(list(clf.staged_decision_function(X_train)))))
sdfTest = 1 / (1 + np.exp(-np.asarray(list(clf.staged_decision_function(X_test)))))
# Сравниваем вероятности с истинными значениями и получаем оценки функции потерь
train_loss = [log_loss(y_train, x) for x in sdfTrain]
test_loss = [log_loss(y_test, x) for x in sdfTest]

# Графики потерь на обучающей и тестовой выборке
plt.figure()
plt.plot(test_loss, 'r', linewidth=2)
plt.plot(train_loss, 'g', linewidth=2)
plt.legend(['test', 'train', 'forest'])
plt.show()

# Если n_estimators несколько сотен, то тут случайный лес работает лучше чем градиентный бустинг
clf = RandomForestClassifier(n_estimators=37, random_state=241)
clf.fit(X_train, y_train)
forest_loss = log_loss(y_test, clf.predict_proba(X_test))
print(forest_loss)