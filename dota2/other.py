import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

features = pd.read_csv('features.csv').drop(columns=['match_id'])

# Разделяем в train входы и выходы
X_features = features.drop(columns=['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                                    'barracks_status_radiant', 'barracks_status_dire'])
# Опция - пробуем убрать категориальные признаки из выборки, а разницы нет
X_features = X_features.drop(columns=['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                                      'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'])
# One Hot кодирование средствами pandas
lobby = pd.get_dummies(features['lobby_type'])
# Преименовыаем столбцы, чтобы избежать совпадений названий столбцов в будущем
lobby = lobby.rename(columns={0: 'zero', 1: 'one', 7: 'seven'})
# Кодируем категориальные признаки в бинарные. Из 10 признаков получаем 108, пользуемся тем, что исходные
# 10 признаков имеют общие возможные значения
# N = pd.unique(features['r1_hero']) — количество различных героев в выборке
heroes = sum([pd.get_dummies(features['r%d_hero' % i]) + -1 * pd.get_dummies(features['d%d_hero' % i])
              for i in range(1, 6)])
# Добавляем признаки lobby и heroes в X_features
X_features = pd.concat([X_features, heroes, lobby], axis=1)

# Возможно здесь достаточно столбца radiant_win. Но интересно попробовать уйти от задачи классификации к задаче
# регресии, если использовать tower_status...
Y_features = features.filter(items=['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                                    'barracks_status_radiant', 'barracks_status_dire'])
# Заполнение пропусков
# Заменяем на нули
X_features.fillna(0, inplace=True)
# Масштабируем признаки
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)


# По результатам работы цикла можно прикинуть важность каждого признака для модели (numbercolumn)
crossvalidator = KFold(n_splits=4, shuffle=True)
for c in [1]:
    break ########################
    start_time = datetime.datetime.now()
    # Параметр регуляризации С на результат влияет очень мало почему-то
    clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=350, max_features=0.2, max_samples=0.2)
    randomnumbers = crossvalidator.split(X=X_features)
    auc = cross_val_score(clf, X_features, Y_features['radiant_win'], cv=crossvalidator,
                          scoring='roc_auc',
                          n_jobs=4)
    print(f"C={c}", "auc =", sum(auc) / len(auc))
    print("Duration:", datetime.datetime.now() - start_time)


# По результатам работы цикла можно прикинуть важность каждого признака для модели (numbercolumn)
crossvalidator = KFold(n_splits=4, shuffle=True)
for c in [1]:
    break  ########################
    start_time = datetime.datetime.now()
    # Параметр регуляризации С на результат влияет очень мало почему-то
    clf = RandomForestClassifier(n_estimators=100, max_depth=3)
    randomnumbers = crossvalidator.split(X=X_features)
    auc = cross_val_score(clf, X_features, Y_features['radiant_win'], cv=crossvalidator,
                          scoring='roc_auc',
                          n_jobs=4)
    print(f"C={c}", "auc =", sum(auc) / len(auc))
    print("Duration:", datetime.datetime.now() - start_time)


# KNeighborsClassifier повис
# Perceptron auc=0.63
# SVC не дождался результата
# BaggingClassifier Работает, но медленно, стоит поиграть с параметрами auc=0.70
# RandomForestClassifier стоит поиграть с параметрами auc=0.66
