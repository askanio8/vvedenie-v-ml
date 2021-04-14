from sklearn.svm import SVC
import pandas as pd
import numpy as np

# Найти опорные вектора в обучающей выборке
data = pd.read_csv('svm-data.csv', header=None)

xtrain = data.drop(columns=0, axis=1)
ytrain = data.filter(items=[0])
ytrain = np.ravel(ytrain)

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(xtrain, ytrain)

a = clf.predict(xtrain.iloc[[3, 4, 9]])  # 4й, 5й и 10й объекты даём на распознавание
a = clf.predict(xtrain)  # Обучающую выборку на распознавание
print(a)
print(clf.support_)  # Список трёх опорных векторов, нумерация с нуля [3 4 9]
