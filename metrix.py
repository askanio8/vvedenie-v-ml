import pandas as pd
from sklearn import metrics

data = pd.read_csv('classification.csv')  # Просто два столбца - истинный класс и его предказание какой-то моделью

TP = len(data[data['true'] == 1][data['pred'] == 1].index)  # Количество верных срабатываний
FP = len(data[data['true'] == 0][data['pred'] == 1].index)  # Количество ложных срабатываний
FN = len(data[data['true'] == 1][data['pred'] == 0].index)  # Количество ложных пропусков
TN = len(data[data['true'] == 0][data['pred'] == 0].index)  # Количество верных пропусков
print(TP, FP, FN, TN)

# Доля верно угаданных - Accuracy (TP + TN) / (TP + FN + FP + TN)
accuracy = metrics.accuracy_score(data.filter(items=['true']), data.filter(items=['pred']))
# Точность - Precision TN / (TN + FP)
precision = metrics.precision_score(data.filter(items=['true']), data.filter(items=['pred']))
# Полнота - Recall TN / (TN + FN)
recall = metrics.recall_score(data.filter(items=['true']), data.filter(items=['pred']))
# F-мера - гармоническое среднее точности и полноты (2 * precision * recall) / (precision + recall)
f1measure = metrics.f1_score(data.filter(items=['true']), data.filter(items=['pred']))
print(accuracy, precision, recall, f1measure)

###################################################################################################################
# истинные классы
# для логистической регрессии — вероятность положительного класса (колонка score_logreg),
# для SVM — отступ от разделяющей поверхности (колонка score_svm),
# для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
# для решающего дерева — доля положительных объектов в листе (колонка score_tree).
data = pd.read_csv('scores.csv')

# Площади под ROC-кривой для каждой модели
logregROC = metrics.roc_auc_score(data.filter(items=['true']), data.filter(items=['score_logreg']))
svmROC = metrics.roc_auc_score(data.filter(items=['true']), data.filter(items=['score_svm']))
knnROC = metrics.roc_auc_score(data.filter(items=['true']), data.filter(items=['score_knn']))
treeROC = metrics.roc_auc_score(data.filter(items=['true']), data.filter(items=['score_tree']))
print(logregROC, svmROC, knnROC, treeROC)
# score_logreg

#  Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70%
logregPRC = metrics.precision_recall_curve(data.filter(items=['true']), data.filter(items=['score_logreg']))
svmPRC = metrics.precision_recall_curve(data.filter(items=['true']), data.filter(items=['score_svm']))
knnPRC = metrics.precision_recall_curve(data.filter(items=['true']), data.filter(items=['score_knn']))
treePRC = metrics.precision_recall_curve(data.filter(items=['true']), data.filter(items=['score_tree']))
# treeROC - 0.65
