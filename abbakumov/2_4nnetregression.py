# Для регрессии в тренде сеть LSTM, а здесь обычная
# Импорт библиотек
import numpy
import pandas as pd

# Импорт matplotlib
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Импорт обучающей выборки
sales = pd.read_csv('monthly-car-sales-in-quebec-1960.csv', sep=';', header=0, parse_dates=[0])

#  Знакомимся с данными
print(sales.head())
print(sales.shape)
print(sales.tail())

# График, чтобы ответить на вопросы о его характеристиках
# Периодичность(сезонность), растущий тренд, отсутствие пропусков, отсутствие явных выбросов, смена характера ряда
# после 25 наблюдений(стало больше сезонных отклонений) их стоит отбросить
sales.iloc[:,1].plot()  # Здесь функция pandas использует matplotlib неявно
plt.show()

# Преобразуем данные
sales_2 = pd.DataFrame()

for i in range(12,0,-1):
    sales_2['t-'+str(i)] = sales.iloc[:,1].shift(i)  # Здесь методом shift сдвигаем стобец вниз на отступ i

sales_2['t'] = sales.iloc[:,1].values  # добавляем несдвинутый столбец, это будет y
# Первые 12 строк теперь с пропусками, их отбрасываем. Вобще, если пропуски в середине, их можно аппроксимировать

sales_4 = sales_2[12:]
#  предикторы и отклик разделяем
# Отклик - группирующая переменная -  вектор y
y = sales_4['t']
# Предикторы - таблица X
X = sales_4.drop('t', axis=1)
# Разделяем на обучающую и тестовую выборки
# Тестовая - последние наблюдения. Брать последние наблюдения в качестве тестовых рекомендует Аббакумов, но можно
# Но можно попробовать взять случайные, только зачем случайные непонятно. Размер тестовой выборки в конце лучше брать
# равным длине переиода прогнозирования
X_train = X[:91]
y_train = y[:91]
X_test  = X[91:]
y_test  = y[91:]
#  Преобразование pandas dataframe в numpy array
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#  Обучение нейронной сети

# Creating a model
model = Sequential()
model.add(Dense(8, input_dim=12, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compiling model
# Минимизируем ошибку по mean_squared_error(если есть выбросы, то возможно лучше mean_absolute_error), а глазами
# оцениваем mean_absolute_percentage_error - ошибка в процентах
# optimizer можно взять и sgd, можно с моментом нестерова
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])

# Training a model
# batch_size=None потому что выборка очень маленькая
model.fit(X_train, y_train, epochs=300, batch_size=None)

# оценка качества модели на тестовом множестве
# Ошибка от 5 до 20% от запуска к запуску
# Так делать неправильно, если нужен прогноз более, чем на 1 шаг. Здесь в оценке мы не учитываем накопление ошибки
# по предыдущим прогнозам.
scores = model.evaluate(X_test, y_test)
print("\nMAPE: %.2f%%" % (scores[1]))

# Вычисляем прогноз
predictions = model.predict(X_test)
# Вычисляем подгонку
predictions_train = model.predict(X_train)
#   График с результатами
x2 = numpy.arange(0, 91, 1)
x3 = numpy.arange(91, 96, 1)
plt.plot(x2, y_train, color='blue')
plt.plot(x2, predictions_train, color='green')
plt.plot(x3, y_test, color='blue')
plt.plot(x3, predictions, color='red')
plt.show()
# В предсказаниях не учтено накопление ошибки!!!