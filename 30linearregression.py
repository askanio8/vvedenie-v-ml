import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge


traindata = pd.read_csv('salary-train.csv')
testdata = pd.read_csv('salary-test-mini.csv')

# Переводим тексты в нижний регистр
traindata['FullDescription'] = traindata['FullDescription'].str.lower()
testdata['FullDescription'] = testdata['FullDescription'].str.lower()

# Заменяем всё, кроме букв и цифр на пробелы для лучшего разбиения на слова
traindata['FullDescription'] = traindata['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
testdata['FullDescription'] = testdata['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

# Из столбца FullDescription делаем тысячи столбцов слов, которые встречаются в более чем 5 объектах
# Их значения - взвешенное среднее (TF-IDF)
# Результат - матрица, а не DataFrame
vectorizer = TfidfVectorizer(min_df=5)
trainFullDescriptionVectorizer = vectorizer.fit_transform(traindata['FullDescription'])
testFullDescriptionVectorizer = vectorizer.transform(testdata['FullDescription'])

# Заполняем nan пропуски значений в столбцах LocationNormalized и ContractTime
traindata['LocationNormalized'].fillna('nan', inplace=True)
testdata['LocationNormalized'].fillna('nan', inplace=True)
traindata['ContractTime'].fillna('nan', inplace=True)
testdata['ContractTime'].fillna('nan', inplace=True)

# Из двух столбцов категориальных признаков LocationNormalized и ContractTime двумерную матрицу -
# для каждого варианта признака выделяем отдельный признак со значением 0 или 1.
# Это называется one-hot-кодирование. Используем DictVectorizer
# Результат двумерная матрица. Строки - объекты, первые три столбца - ContractTime, последние 1763 столбца - локации
# Если в конструктор передать sparse=False, то результатом будет не scipy.sparse.csr_matrix а numpy.ndarray
# в ndarray лучше предпросмотр
dvectorizer = DictVectorizer()  # можно передать sparse=False
# Строка ниже переводит два столбца в словарь вида: строка->{LocationNormalized:London, ContractTime: permanent}...
# argument = traindata[['LocationNormalized', 'ContractTime']].to_dict('records')
trainLocationContractVecrorizer = dvectorizer.fit_transform(traindata[['LocationNormalized',
                                                                       'ContractTime']].to_dict('records'))
testLocationContractVecrorizer = dvectorizer.transform(testdata[['LocationNormalized',
                                                                 'ContractTime']].to_dict('records'))

# Объединяем матрицы признаков
trainMatrix = hstack([trainFullDescriptionVectorizer, trainLocationContractVecrorizer])
testMatrix = hstack([testFullDescriptionVectorizer, testLocationContractVecrorizer])

# Обучаем, предсказываем
clf = Ridge(alpha=1, random_state=241)  # alpha - сила регуляризации, 1/2*С в других моделях
clf.fit(trainMatrix, traindata.filter(items=['SalaryNormalized']))
result = clf.predict(testMatrix)
print(result)
