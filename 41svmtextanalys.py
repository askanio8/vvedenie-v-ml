import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# После выполнения этого кода массив с текстами будет находиться в поле newsgroups.data,
# номер класса — в поле newsgroups.target.
newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

#  Возвращает TF-IDF(частота встречаемости слова в тексте/количество текстов с этим словом) примерно так
#  Значения могут быть отрицательными в случе двух классов(в этой зачдаче по крайней мере)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data, newsgroups.target)

# Параметр 'C' - ширина разделяющей в SVM
grid = {'C': np.power(10.0, np.arange(-5, 6))}  # np.power возводит 10.0 в степени от -5 до 5
# Здесь первым параметром раньше передавал длину выборки, но вроде бы это не обязательный параметр
# Есть идея использовать количество уникальных слов len(vectorizer.get_feature_names()) а оно и так работает
cv = KFold(n_splits=5, shuffle=True, random_state=241)  # Делит выборку на 5 частей для кросс-валидации
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)  # Подбирает наилучший параметр 'C' для классификатора clf
gs.fit(X, newsgroups.target)
C = gs.best_params_['C']  # Найденное значение параметра ширины разделяющей полосы 1.0

clf = SVC(kernel='linear', random_state=241, C=C)  # Создаём новый SVM с параметром С
clf.fit(X, newsgroups.target)

# Находим самые важные слова для классификации
z = clf.coef_.toarray()  # Разреженная матрица в массив numpy
z = np.abs(z)  # Все значения по модулю в этом случае
maxwordsindexes = []
for i in range(10):
    maxwordsindexes.append(np.argmax(z[0]))
    z[0, np.argmax(z)] = 0
maxwords = [vectorizer.get_feature_names()[i] for i in maxwordsindexes]
print(sorted(maxwords))


# Еще один метод
result_test = pd.DataFrame(clf.coef_.todense())  # Уплотняет разреженную матрицу со значениями 0
result_test = result_test.abs()
result_test = result_test.sort_values(0, axis=1)
result_test = result_test.iloc[:, -10:].columns