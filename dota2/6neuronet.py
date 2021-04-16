import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

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
for с in [1]:
    start_time = datetime.datetime.now()
    # activation="logistic" здесь чуть лучше а "relu" должна быть быстрее
    # solver='adam' лучше на одной итерации, а 'sgd' улучшает результат на нескольких. ‘lbfgs’ повис
    # lbfgs рекомендуют для маленького датасета
    # alpha=0.1 коэф регуляризации l2 ухудшает
    # batch_size=1100 - накопитель поправок весов как я понял. разница очень маленькая
    # learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, толко с solver='sgd'
    # learning_rate_init
    # tol=0.0001 порог изменения качества? для прекращения обучения, альтернатива max_iter
    # early_stopping=True должно быть True иначе tol не подействует
    # validation_fraction=0.1 выделить долю выборки для валидации для ранней остановки early_stopping
    clf = MLPClassifier(verbose=True, hidden_layer_sizes=(2000), max_iter=14, activation="logistic",
                        solver="adam", batch_size=1100, learning_rate_init=0.0001, tol=0.0001,
                        early_stopping=True, validation_fraction=0.1)
    randomnumbers = crossvalidator.split(X=X_features)
    auc = cross_val_score(clf, X_features, Y_features['radiant_win'], cv=crossvalidator,
                          scoring='roc_auc',
                          n_jobs=4)
    print(f"c=", "auc =", sum(auc) / len(auc))
    print("Duration:", datetime.datetime.now() - start_time)

# Результат Auc=0.75 как и у других методов