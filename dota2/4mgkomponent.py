import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

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

# Метод главных компонент
pca = PCA(n_components=202)
pca.fit(X_features)
# Процент дисперсии, который объясняет каждая компонента
print(pca.explained_variance_ratio_)  # Первые 4 компоненты отвечают за больше чем ... дисперсии
# Процент вкалада каждого исходного признака в каждую результирующую компоненту
print(pca.components_)
# Признак с самым большим вкладом в первую компоненту
print(np.argmax(pca.components_[0]))
# Преобразованные признаки
X_features = pca.transform(X_features)

# По результатам работы цикла можно прикинуть важность каждого признака для модели (numbercolumn)
crossvalidator = KFold(n_splits=5, shuffle=True)
for numbercolumn in range(0, X_features.shape[1]):
    X_featuresWithoutOneColumn = np.delete(X_features, numbercolumn, 1)
    start_time = datetime.datetime.now()
    # Параметр регуляризации С на результат влияет очень мало почему-то
    clf = LogisticRegression(penalty='l2', C=1)
    randomnumbers = crossvalidator.split(X=X_featuresWithoutOneColumn)
    auc = cross_val_score(clf, X_featuresWithoutOneColumn, Y_features['radiant_win'], cv=crossvalidator,
                          scoring='roc_auc',
                          n_jobs=4)
    print(f"numbercolumn={numbercolumn}", "auc =", sum(auc) / len(auc))


# 1. Метод главных компонент не улучшил результат обучения, скорее всего нет сильной корреляции между признаками
# Сам метод уменьшает размерность данных, при этом плавно снижается уровень обучения
# Есть смысл применять если очень много признаков и возможно есть коррелирующие
# 2. Пробовал удалять каждый признак по отдельности. Обычно уровень обучения снижался в пределах одной десятой
# процента, или не снижался вообще
# 3. Удалил 20 случайных признаков из самых малозначимых auc снизился на одну деятую процента.
# Алгоритм просто не принимает во внимание признаки если они не коррелируют с выходом
