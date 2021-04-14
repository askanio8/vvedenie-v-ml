import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("perceptron-train.csv", header=None)
test_data = pd.read_csv("perceptron-test.csv", header=None)

xtrain = train_data.drop(columns=0, axis=1)
ytrain = train_data.filter(items=[0])
ytrain = np.ravel(ytrain)  # Столбец в строку

xtest = test_data.drop(columns=0, axis=1)
ytest = test_data.filter(items=[0])
ytest = np.ravel(ytest)  # Столбец в строку

scaler = StandardScaler()  # Нормализация признаков
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

clf = Perceptron(n_jobs=4)
clf.fit(xtrain, ytrain)
predictions = clf.predict(xtest)
acccuracy = accuracy_score(ytest, predictions)

print(acccuracy)  # 0.655 без нормализации признаков  0.845 с нормализацией