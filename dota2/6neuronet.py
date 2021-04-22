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


############Вставка tensorflow№№№№№№№№№№№№№№№№№№№№№№№№№№№№
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features, Y_features['radiant_win'].values,
                                                    random_state=12345,
                                                    # доля объёма тестового множества
                                                    test_size=0.33)
y_train_bin = utils.to_categorical(y_train)
y_test_bin = utils.to_categorical(y_test)

# Creating a model
model = Sequential()
model.add(Dense(230, input_dim=202, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train_bin, epochs=7, batch_size=1000)
# evaluate the model
scores = model.evaluate(X_test, y_test_bin)
print("\nAccuracy: %.2f%%" % (scores[1]*100))
# tensorflow точно быстрее и кажется лучше качество
# Но здесь явно нужно следить за переобучением
###########################################################################




# По результатам работы цикла можно прикинуть важность каждого признака для модели (numbercolumn)
crossvalidator = KFold(n_splits=4, shuffle=True)
for act in ['identity', 'logistic', 'tanh', 'relu']:
    start_time = datetime.datetime.now()
    # activation="logistic" здесь чуть лучше а "relu" должна быть быстрее
    # solver='adam' стоит начать с этого метода и он здесь лучше на одной итерации, он уменьшает шаг при большой
    # прозводной, 'sgd' здесь продолжает улучшать результат на большем количестве эпох, ‘lbfgs’ повис
    # momentum - только для 'sgd', во время градиентного спуска каждый следующий шаг увеличивается если последние
    # несколько шагов направлены в одну сторону. Это дополнительное слагаемое в формуле градиентного спуска
    # Рекомендуют начать с 0.9. Если 0, то это обычный градиентный спуск, если больше 0.9, то метод энергичнее
    # перескакивает локальные(может и глобальный кстати) минимумы
    # nesterov momentum - усовершенствование momentum, не запоминается динна предыдущих шагов, а только их направление
    # и частота смены направления. Может быть лучше adam в некоторых задачах
    # alpha - коэф регуляризации l2. Если adam, то 0.1 ухудшает
    # batch_size=1100 - накопитель поправок весов как я понял. разница очень маленькая. Чем больше значение, тем
    # быстее должно быть обучение на одной эпохе, а маленькие значения могут помочь избежать локальных минимумов
    # learning_rate_init - Коэф скорости обучения
    # learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, толко с solver='sgd' - изменение learning_rate_init
    # со временем. ‘constant’ - не изменяется. ‘invscaling’ уменьшает,используя параметр power_t
    # по формуле effective_learning_rate = learning_rate_init / pow(t, power_t). ‘adaptive’ уменьшает эсли между
    # эпохами потери не умньшились хотя бы на параметр tol если 'early_stopping' включен, текущая скорость обучения
    # делится на 5.
    # tol=0.0001 порог изменения качества? для прекращения обучения, альтернатива max_iter
    # early_stopping=True должно быть True иначе tol не подействует
    # validation_fraction=0.1 выделить долю выборки для валидации для ранней остановки early_stopping
    # warm_start - после завершения обучения можно поменять параметры и продолжить дообучать модель

    # Это adam
    #clf = MLPClassifier(verbose=False, hidden_layer_sizes=(1300,800,600,500), max_iter=4, activation=act,
    #                    solver="adam", batch_size=1100, learning_rate_init=0.0001, tol=0.0001,
    #                    early_stopping=True, validation_fraction=0.1,)
    # Это sgd
    # Здесь с функцией активации logistic очень плохо если большое значение batch_size, видимо попадает в лок минимум
    # Зато большой batch_size намного ускоряет скорость обучения на эпоху
    clf = MLPClassifier(verbose=False, hidden_layer_sizes=(230,80,60,50), max_iter=10, activation=act,
                        solver='sgd', learning_rate_init=0.01, batch_size=1000, tol=0.1, learning_rate='adaptive')
                        #early_stopping=True, validation_fraction=0.1,)
    randomnumbers = crossvalidator.split(X=X_features)
    auc = cross_val_score(clf, X_features, Y_features['radiant_win'], cv=crossvalidator,
                          scoring='roc_auc',
                          n_jobs=4)
    print(f"act={act}", "auc =", sum(auc) / len(auc))
    print("Duration:", datetime.datetime.now() - start_time)

# Результат Auc=0.75 как и у других методов