import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import datetime

features = pd.read_csv('features.csv', index_col='match_id')
features_test = pd.read_csv('features_test.csv', index_col='match_id')

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

crossvalidator = KFold(n_splits=5, shuffle=True)
for c in [0.001, 0.01, 0.1, 1, 10, 100]:
    start_time = datetime.datetime.now()
    # Параметр регуляризации С на результат влияет очень мало почему-то
    clf = LogisticRegression(penalty='l2', C=c)
    randomnumbers = crossvalidator.split(X=X_features)
    auc = cross_val_score(clf, X_features, Y_features['radiant_win'], cv=crossvalidator, scoring='roc_auc',
                          n_jobs=4)
    print(f"C={c}", "auc =", sum(auc) / len(auc))
    print("Duration:", datetime.datetime.now() - start_time)

# Работа с тестовой выборкой
X_test = features_test.drop(columns=['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                                     'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'])
lobby = pd.get_dummies(features_test['lobby_type'])
lobby = lobby.rename(columns={0: 'zero', 1: 'one', 7: 'seven'})
heroes = sum([pd.get_dummies(features_test['r%d_hero' % i]) + -1 * pd.get_dummies(features_test['d%d_hero' % i])
              for i in range(1, 6)])
X_test = pd.concat([X_test, heroes, lobby], axis=1)
X_test.fillna(0, inplace=True)
X_test = scaler.transform(X_test)
clf = LogisticRegression(penalty='l2', C=0.1, n_jobs=4)
clf.fit(X_features, Y_features['radiant_win'])
res = clf.predict_proba(X_test)
print('max:', max(res[0]), 'min:', min(res[0]))  # Результат

######################################ОТЧЕТ##############################
# 1. Результат логистической регрессии AUC=0.716. Разницы с градиентным бустингом практически нет, возможно
# без других предобработок данных этот результат возле максимума. Скорость работы логистической регрессии
# на порядок выше градиентного бустинга.
# 2. После удаления категориальных признаков качество обучения не изменилось, возможно веса этих признаков в
# результате обучения становились близкими к 0.
# 3. Всего 108 разных героев.
# 4. Добавление мешка слов улучшило качество, теперь AUC = 0.752. Скорее всего некоторые герои или сочетания
# героев сильнее других.
# 5. Максимальное значение вероятности - 0.815, минимальное - 0.184


# Еще
# 1. Roc-auc
# 2. Анализ выбросов
# 3. Анализ признаков
# 4. Другие методы