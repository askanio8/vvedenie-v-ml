# Импорт библиотек
import numpy
import pandas as pd

# Импорт matplotlib - не нужен, на всякий случай
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from tensorflow.keras import initializers

wine = pd.read_csv('Wine.txt', sep='\t', header=0)
print(wine.head())

# Смотрим баланс классов в выборке
balans = wine['Desired1(3)'].value_counts(normalize=True)  # normalize=True означает в долях
print(balans)

#  предикторы и отклик разделяем
# Отклик - группирующая переменная -  вектор y
y = wine['Desired1(3)']
# Предикторы - таблица X
X = wine.drop('Desired1(3)', axis=1)

# Странно что здесь для разделения выборки используем sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12345,
                                                    # доля объёма тестового множества
                                                    test_size=0.33)

#  Преобразование pandas dataframe в numpy array
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# Больше 2-х классов. Классы не упорядочены
# Делаем из одного выходного столбца с номерами классов три столбца с бинарными метками классов
y_train_bin = utils.to_categorical(y_train)
y_test_bin = utils.to_categorical(y_test)

# Инициализация весов усеченным нормальным распределением. Если сеть небольшая зерно можно хранить вместо сети
init_2 = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=12345)
# Инициализация свободных членов
init_3 = initializers.Constant(value = 1e-3)
# Creating a model
model = Sequential()
model.add(Dense(90, input_dim=13, activation='relu'))
# Пример инициализации весов, для каждого слоя может быть разный метод
model.add(Dense(10, activation='relu', kernel_initializer=init_2, bias_initializer=init_3))
# Последний слой три нерона, потому что у нас три класса. а softmax нам показывает псевдовероятности
# Если задача регрессии, то вместо logistic или softmax линейная активационная функция
model.add(Dense(3, activation='softmax'))

# Compiling model
# Здесь происходит минимизация функции потерь categorical_crossentropy, а резултаты получаем таки accuracy!
# categorical_crossentropy использучеся для нескольких неупорядоченых классов
# sparse_categorical_crossentropy Аббакумов рекомендует только если классы упорядочены
# Для двух классов есть binary_crossentropy(похоже на функцию правдоподобия) и ещё несколько
# kl_divergence function только для сетей Variational auto encoders
# poisson функция Пуассона используется для предсказания количества? вероятностей? какого-то события, только при
# малых вероятностях
# Для задач регресии используются mean_squared_error function
# mean_absolute_error function - сумма модулей ошибок, больше игнорирует выбросы чем mean_squared_error
# mean_absolute_percentage_error function - эта ошибка измеряется в % от самого y
# mean_squared_logarithmic_error function - как и mean_absolute_error меньше обращает внимания на выбросы
# cosine_similarity function
# log_cosh - гибрид, ведет себя как mean_squared_error, если ошибка небольшая, иначе как mean_absolute_error
# и др.
# Еще одна категория функций потерь - Hinge loss. она больше обращает внимание на пограничные объекты
# Compiling model
#sgd2 = optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.8, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd2, metrics=['accuracy'])
# Здесь с adam намного лучше, стоит поиграть с параметрами sgd
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training a model
# Процент ошибок разнонаправленно меняется от эпохи к эпохе
# Очень разные результаты после каждого выполнения, тут очень выжно зерно случайности
model.fit(X_train, y_train_bin, epochs=100, batch_size=10)

# evaluate the model
scores = model.evaluate(X_test, y_test_bin)
print("\nAccuracy: %.2f%%" % (scores[1]*100))

# calculate predictions
predictions = model.predict(X_test)
# round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)
predictions