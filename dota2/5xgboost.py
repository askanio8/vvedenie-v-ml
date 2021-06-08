import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import xgboost
import matplotlib
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV  # Для калибровки

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

crossvalidator = KFold(n_splits=4, shuffle=True)
for c in [500]:
    start_time = datetime.datetime.now()
    # Если ограничить глубину дерева max_depth, то алгоритм работает лучше рекомендуют 3-10
    # max_leaf_nodes преоределит max_depth как 2**max_depth
    # colsample_bytree не заметил разницы рекомендуется 0.3-0.8, 1
    # learning_rate 0.01 0.2 падает уровень, лучше 0.1
    # subsample рекомендуют 0.8-1 разницы не заметил
    # gamma - регуляризация, рекомендуют 0, 1, 5, тут лучше 8-15
    # reg_alpha=5, reg_lambda=5
    # scale_pos_weight рекомендуют больше при несбалансированности классов, здесь 2 лучше
    # base_score порог, по умолчанию 0.5
    # booster = gbtree Может быть ансамбль деревьев или линейных моделей. Деревья лучше
    # reg_alpha, reg_lambda - L1, L2 регуляризация

    clf = xgboost.XGBClassifier(
                                base_score=0.5,
                                colsample_bylevel=1, # есть мнение что лучше не трогать, замедляет
                                colsample_bytree=0.66,  # 0.5-1 или sqrt(numcolumns)
                                gamma=0.01,  # коэф мин допустимого уменьшения загрязненности при расщеплении. подбирать
                                learning_rate=0.1, # рекомендуют 0.01-0.2
                                max_delta_step=0,  # Аббакумов не знает что это
                                max_depth=4,  # нужно подбирать. рекомендуют <6
                                min_child_weight=5,  # мин объектов в узле. стоит пробовать больше 5...10...
                                missing=None,
                                n_estimators=c,  # подбирать
                                nthread=1,  # ?? наверное для нескольких компютеров
                                n_jobs=1,
                                objective='binary:logistic',
                                reg_alpha=5,  # коэф регуляризации L1  подбирать
                                reg_lambda=0,  # коэф регуляризации L2
                                scale_pos_weight=1,  # при несбалансированности выборки
                                seed=0,
                                subsample=0.8,  # подбирать. рекомендуют 0.66
                                verbosity=1,
                                )
    #model = clf.fit(X_features, Y_features['radiant_win'],)
    #xgboost.plot_importance(model)  #  Важность параметров
    #print(model.feature_importances_)

    randomnumbers = crossvalidator.split(X=X_features)
    auc = cross_val_score(clf, X_features, Y_features['radiant_win'], cv=crossvalidator,
                          scoring='roc_auc',
                          n_jobs=4)
    print(f"numbercolumn={c}", "auc =", sum(auc) / len(auc))
    print("Duration:", datetime.datetime.now() - start_time)

# Заметил переобучение в xgboost при количестве деревьев 1000, 3000
# С этими параметрами auc=0.75, так же как и у Adaboost, возможно гиперпараметры еще не подобраны
# С масштабироваием признаков и без результат тот же
# XGBoost может работать с пропусками. Пропуски воспринимаются как информация и в зависимости от их
# появления строится прогноз. Если в появлении пропусков нет зависимости от класса обьекта, то возможно
# лучше убрать объекты с пропусками из обучающей выборки
