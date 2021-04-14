import pandas as pd

data = pd.read_csv('titanic.csv', index_col='PassengerId')

# print(data[:10])  # 10 верхних строк
# print(data.head())  # 5 верхних строк

print(data['Pclass'].value_counts())  # Статистика столбца

print(data.columns.values.tolist())  # Список столбцов
# 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
print(data['Sex'].value_counts())  # Количество мужчин и женщин на корабле 577 314
# 22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
dataS = data['Survived'].value_counts()
x = float(pd.Series(data=dataS, index=[1]))  # Количество выживших
print("{:.2f}".format(x * 100 / data.shape[0]))  # Процент выживших 38.38
# 33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
dataC = data['Pclass'].value_counts()
x = float(pd.Series(data=dataC, index=[1]))  # Количество пассажиров 1 класса
print("{:.2f}".format(x * 100 / data.shape[0]))  # Процент пассажиров 1 класса 24.24
# 44444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444
# group = data['Age'].sort_values()  # Сортировать не обязательно
print("{:.2f}".format(data['Age'].mean()))  # Средний возраст 29.70
print(data['Age'].median())  # Возраст в середине списка - медиана 28.0
# 55555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555
x = data.filter(items=['SibSp', 'Parch'])  # Фильтр по столбцам
pearson = x.corr(method='pearson')['SibSp'].min()  # Коэф корелляции Пирсона 0.41
print("{:.2f}".format(pearson))
# 666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666
dataN = data.filter(items=['Name', 'Sex'])  # Оставляем только 2 столбца
dataN = dataN[dataN['Sex'] == 'female']  # Оставляем только сторки, в столбце Sex которых female
listnames = dataN['Name'].tolist()  # Столбец в список
names = []
err = []
for strname in listnames:
    if strname.find('(') != -1:  # Если есть скобки, то имена в скобках
        s = strname.split('(')
        s = s[1].strip(') ')
        names += s.split()
    elif strname.find('Miss.') != -1:  # Если в строке есть слово Miss. ищем имена за ним
        s = strname.split('Miss.')
        s = s[1].strip(') ')
        names += s.split()
    else:
        err.append(strname)  # Нестандартные записи кидаем в err

comname = ''
for x in names:
    if names.count(x) > names.count(comname):
        comname = x
print(comname)  # Самое популярное женское имя - Mary
