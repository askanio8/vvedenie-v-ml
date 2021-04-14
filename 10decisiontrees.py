import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('titanic.csv', index_col='PassengerId')

# Найтии зависимость выживания от класса, пола, возраста и цены билета
dataN = data.filter(items=['Survived', 'Pclass', 'Sex', 'Age', 'Fare'])  # Оставляем 5 столбцов
dataN = dataN[dataN['Age'].notnull()]  # Оставляем только сторки, в которых указан возраст
dataX = dataN.filter(items=['Pclass', 'Sex', 'Age', 'Fare'])  # Входные значения
dataY = dataN.filter(items=['Survived'])  # Выходные значения
mapping = {'male': 0, 'female': 1}  # Словарь для замены текста на число
dataX = dataX.replace(mapping)  # Замена

clf = DecisionTreeClassifier(random_state=241)  # В данном случае зерно случайности мало влияет на результат обучения
clf.fit(dataX, dataY)  # Обучение
importances = clf.feature_importances_  # Показывает важность признаков

outs = clf.predict([[3, 0, 10, 100]])  # Это предсказывает! 0 или 1

print(outs)
print(importances)
######################################ОЦЕНКА ТОЧНОСТИ ОБУЧЕНИЯ######################################################
dataN = data.filter(items=['Survived', 'Pclass', 'Sex', 'Age', 'Fare'])  # Оставляем 5 столбцов
dataN = dataN[dataN['Age'].notnull()]  # Оставляем только сторки, в которых указан возраст
dataN = dataN.sample(frac=1)  # Перемешиваем На всякий случай
dataTest = dataN[614:]  # Отделяем тестовую выборку 100 строк
dataN = dataN[:614]  # Обучающая выборка
print(dataN)

dataX = dataN.filter(items=['Pclass', 'Sex', 'Age', 'Fare'])  # Входные значения
dataY = dataN.filter(items=['Survived'])  # Выходные значения
mapping = {'male': 0, 'female': 1}  # Словарь для замены текста на число
dataX = dataX.replace(mapping)  # Замена

clf = DecisionTreeClassifier(random_state=241)  # В данном случае зерно случайности мало влияет на результат обучения
clf.fit(dataX, dataY)  # Обучение

mapping = {'male': 0, 'female': 1}  # Словарь для замены текста на число
dataTest = dataTest.replace(mapping)  # Замена
outs = clf.predict(dataTest.filter(items=['Pclass', 'Sex', 'Age', 'Fare']))  # Скармливаем тестовую выборку
outTest = dataTest['Survived'].tolist()
per = 0
for i, b in enumerate(outs):
    if outTest[i] == b:
        per += 1
print(per)  # Точность тут 75-80%, а на обучающих данных 99-100
