import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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

# Cтолбец radiant_win.
Y_features = features.filter(items=['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                                    'barracks_status_radiant', 'barracks_status_dire'])
# Заполнение пропусков
# Заменяем на нули
X_features.fillna(0, inplace=True)
# Масштабируем признаки
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)

##################Adaboost как детектор выбросов#################
auc = []
for n in range(5):
    # Делим выборку на 2 части
    X_train, X_test, y_train, y_test = train_test_split(X_features, Y_features, test_size=0.2)
    # Ищем выбросы
    clf = AdaBoostClassifier(n_estimators=30, learning_rate=0.1)
    clf.fit(X_train, y_train['radiant_win'])
    res = clf.predict_proba(X_train)[:, 1]
    z = y_train['radiant_win'].values - res
    z = np.abs(z)
    count = [c[0] for c in enumerate(z) if c[1] > 0.523]  # Список индексов выбросов
    # Выбрасываем выбросы
    X_train = np.delete(X_train, count, axis=0)
    y_train = np.delete(y_train['radiant_win'].values, count, axis=0)

    # Обучаем линейную регрессию
    clf = LogisticRegression(penalty='l2', C=1)
    clf.fit(X_train, y_train)
    res = clf.predict_proba(X_test)
    auc.append(roc_auc_score(y_test['radiant_win'], res[:, 1]))  # Здесь auc на тестовой выборке больше 0.754,
    print(sum(auc)/len(auc))


####################Adaboost сам по себе##############################
crossvalidator = KFold(n_splits=4, shuffle=True)
randomnumbers = crossvalidator.split(X=X_features).__next__()

X_train = X_features[randomnumbers[0]]
Y_train = Y_features['radiant_win'][randomnumbers[0]].values
X_test = X_features[randomnumbers[1]]
Y_test = Y_features['radiant_win'][randomnumbers[1]].values
start_time = datetime.datetime.now()
clf = AdaBoostClassifier(n_estimators=300, learning_rate=0.1)  # При 1000 auc=74, при 3000 74.67
clf.fit(X_train, Y_train)
res = clf.predict_proba(X_test)
auc = roc_auc_score(Y_test, res[:, 1])
acc = accuracy_score(Y_test, clf.predict(X_test))
print("auc =", auc, "acc =", acc)
print("Duration:", datetime.datetime.now() - start_time)

# Результаты обучения без выбросов на две десятые лучше, чем обычно
# Результат незначителен, так как выборка уже была очищена от выбросов при составлении задачи
# Для подготовки сырых данных метод будет работать хорошо
