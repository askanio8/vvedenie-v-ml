# Регрессия с мультипликативной сезонностью. В добавление к растущему тренду растет размах сезонных колебаний
# Такой график тоже можно подавать на входы нейросети, но лучше прологарифмировать чтобы избавиться от изменения
# размаха со временем

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import mape
import random

############################################ПРЕДОБРАБОТКА#################################################
# Импорт обучающей выборки
data = pd.read_csv('series_g.csv', sep=';', header=0)  # 0 - индекс первой строки
# График, чтобы ответить на вопросы о его характеристиках
# Периодичность(сезонность), растущий тренд, отсутствие пропусков, отсутствие явных выбросов, смена характера ряда
fig = plt.figure(figsize=(12, 4))  # Окно для графиков со стартовыми размерами
ax1 = fig.add_subplot(1, 2, 1)  # 1X2 сетка для графиков, этот график в 1 ячейке
data['series_g'].plot(ax=ax1)
ax1.set_title(u'Объём пассажироперевозок')
ax1.set_ylabel(u'Тысяч человек')

#  Надо прогнозировать логарифм. Логарифм от количества тысяч а не от тысяч. Думаю нет разницы для сети
data['log_y'] = np.log10(data['series_g']) # Добавляем столбец с логарифмированными данными
ax2 = fig.add_subplot(1, 2, 2)  # 1X2 сетка для графиков, этот график во 2 ячейке
pd.Series(data['log_y']).plot(ax=ax2)
ax2.set_title(u'log10 от объёма пассажироперевозок')
ax2.set_ylabel(u'log10 от тысяч человек')
plt.show()

# Преобразуем данные в таблицу
datatable = pd.DataFrame()
for i in range(12,0,-1):
    datatable['t-' + str(i)] = data.iloc[:, 2].shift(i)
datatable['t'] = data.iloc[:, 2].values
datatable = datatable[12:]  # Отрезаем первые 12 строк

# Разделяем на обучающую и тестовую выборки
X_train = datatable.drop('t', axis=1)[:120].values
y_train = datatable['t'][:120].values
# Тестовая - последние наблюдения
X_test  = datatable.drop('t', axis=1)[120:].values
y_test  = datatable['t'][120:].values

############################################ПОСТРОЕНИЕ МОДЕЛИ####################################################
#  Обучение нейронной сети
# Creating a model
seed = random.randrange(1000)  # seed = random.randrange(1000); np.random.seed(a) # тоже самое вроде бы
tensorflow.random.set_seed(seed=seed)
model = Sequential()
# Аббакумов рекомендует менше 8 нейронов, мне кажется 800 явно лучше
model.add(Dense(8, input_dim=12, activation='relu'))
# Если много нейронов, слоёв будет переобучение
model.add(Dense(1, activation='linear'))
# Compiling model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
# Training a model
model.fit(X_train, y_train, epochs=5000, batch_size=None, workers=4, use_multiprocessing=True)

# Убираем последние 12 значений из тестовой выборки, будем пошагово добавлять новые значения предказаний чтобы
# предказание строилось на нексколько шагов вперед а не на один. Так ошибка будет накапливаться с каждым
# предсказанием.
predictions_test = []
for i, x in enumerate(X_test):
    # Соединяем значения из конца выборки train и предсказанные
    a = np.append(x[0:len(x)-i], predictions_test).reshape((1, 12))
    # И снова подаём в модель для предказания следующего значения. Сохраняем предсказания в список
    predictions_test.append(model.predict(a)[0, 0])

err = mape(y_test, predictions_test)  # np.mean(100 * abs(X_test[-1] - predictions) / X_test[-1])
print("MAPE с накоплением ошибки: %.2f%%" % err)

# Делаем предсказания моделью для обучающих данных
predictions_train = model.predict(X_train)

#   График с результатами
# Возвращаем реальные значения из логарифмированных
plt.plot(np.arange(0, 120, 1), 10**y_train, color='blue')
plt.plot(np.arange(0, 120, 1), 10**predictions_train, color='green')
plt.plot(np.arange(120, 132, 1), 10**y_test, color='blue')
plt.plot(np.arange(120, 132, 1), 10**np.array(predictions_test), color='red')
plt.show()

############################################ПРЕДСКАЗАНИЕ#################################################
# Если прогноз хороший, сохранить seed на всякий случай и обучить модель на полных данных
model.fit(datatable.drop('t', axis=1).values, datatable['t'].values, epochs=5000, batch_size=None,
          workers=4, use_multiprocessing=True)

predictions_future = []
for i in range(12, 0, -1):
    # Соединяем значения из конца выборки и предсказанные
    a = np.append(y_test[-i:], predictions_future).reshape((1, 12))
    # И снова подаём в модель для предказания следующего значения. Сохраняем предсказания в список
    predictions_future.append(model.predict(a)[0, 0])

#   График с результатами
plt.plot(np.arange(0, 120, 1), 10**y_train, color='blue')
plt.plot(np.arange(0, 120, 1), 10**predictions_train, color='green')
plt.plot(np.arange(120, 132, 1), 10**y_test, color='blue')
plt.plot(np.arange(120, 132, 1), 10**np.array(predictions_test), color='red')
plt.plot(np.arange(132, 144, 1), 10**np.array(predictions_future), color='black')
plt.show()

print("Предсказания:", [round(x, 2) for x in predictions_future])

# Есть идея построить усредненный график скользящим окном шириной 12
# чтобы избавиться от сезонности (это один из методов)
# Исходный ряд минус усредненный график это аддитивная модель(нет погодового растущего тренда)
# Исходный ряд разделить на усредненный график это мультипликативная модель(есть растущий тренд)
# Строим модель по двум рядам
# Еще есть идея стекинга по отобранным лучшим моделям(можно не только нейросети)
# Стоит попробовать слои LSTM, вроде бы они лучше подходят для регрессии
