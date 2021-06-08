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

# С помощью graphviz можно визуализировать дерево
# Загрязненность узла измеряется энтропией p*log2(p) или индексом Джини p*(1-p)
# criterion="gini" или "entropy" - Загрязненность узла, лучше пробовать и то и то
# max_depth - максимальная глубина дерева, нужно подбирать
# min_samples_split - важный параметр
# min_samples_leaf - важный параметр
# max_leaf_nodes - важный параметр
# min_impurity_decrease - минимальное уменьшение загрязнения при расщеплении
# class_weight - при несбалансироавнности классов
# splitter = "best" или "random" возможно random если это лес
# max_features  если это лес, ставим ограниение
# подбор параметров с помощью GridSearch или вручную
# Популярные алгоритмы построения деревьев - CART и C4.5
# Вроде бы деревья в sklearn не могут работать с пропусками, стоит посмотреть в другом пакете
# Так же sklearn не работает с номинальной шкалой признаков, нужно one-hot-encoder(numpy, быстрее в продакшене)
# или get_dummies(pandas)
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

# Деревья могут использоваться в регрессии, расщеплением они строят кусоно-постоянную фунцию, усредняя
# выходные значения суммы объектов в каждом узле
