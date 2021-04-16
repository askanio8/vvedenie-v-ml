import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import datetime


features = pd.read_csv('features.csv', index_col='match_id')
features_test = pd.read_csv('features_test.csv', index_col='match_id')

# Разделяем в train входы и выходы
X_features = features.drop(columns=['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                                    'barracks_status_radiant', 'barracks_status_dire'])
# Достаточно столбца radiant_win.
Y_features = features.filter(items=['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                                    'barracks_status_radiant', 'barracks_status_dire'])
# Теперь количество столбцовв X_features и features_test совпадает
print(features.shape)
print(features_test.shape)

# Проверяем количество пропусков в каждом столбце
count = X_features.count()
print(count)

# Заполнение пропусков
# В случае, если в продакшене гарантируется получение всех признаков и пропусков не будет, а объектов в
# тренировочной выборке достаточно много, то можно просто выбросить объекты с пропусками
# 1. Замените пропуски на нули с помощью функции fillna(). На самом деле этот способ является предпочтительным
# для логистической регрессии, поскольку он позволит пропущенному значению не вносить никакого вклада в
# предсказание.
# Заменяем на нули
X_features.fillna(0, inplace=True)  # Все три варианта примерно на одном уровне
# 2. Для деревьев часто лучшим вариантом оказывается замена пропуска на очень большое или очень маленькое
# значение — в этом случае при построении разбиения вершины можно будет отправить объекты с пропусками в
# отдельную ветвь дерева.
# Замена пропуска на очень большое значение
# X_features.fillna(99999, inplace=True)  # Вроде бы есть маленькая разница в пользу этого варианта
# 3. Также есть и другие подходы — например, замена пропуска на среднее значение признака.
# Замена пропуска на среднее значение признака  # И здесь неплохо
# X_features['first_blood_time'].fillna(X_features['first_blood_time'].mean(), inplace=True)
# X_features['first_blood_team'].fillna(X_features['first_blood_team'].mean(), inplace=True)
# X_features['first_blood_player1'].fillna(X_features['first_blood_player1'].mean(), inplace=True)
# X_features['first_blood_player2'].fillna(X_features['first_blood_player2'].mean(), inplace=True)
# X_features['radiant_bottle_time'].fillna(X_features['radiant_bottle_time'].mean(), inplace=True)
# X_features['radiant_courier_time'].fillna(X_features['radiant_courier_time'].mean(), inplace=True)
# X_features['radiant_flying_courier_time'].fillna(X_features['radiant_flying_courier_time'].mean(), inplace=True)
# X_features['radiant_first_ward_time'].fillna(X_features['radiant_first_ward_time'].mean(), inplace=True)
# X_features['dire_bottle_time'].fillna(X_features['dire_bottle_time'].mean(), inplace=True)
# X_features['dire_courier_time'].fillna(X_features['dire_courier_time'].mean(), inplace=True)
# X_features['dire_flying_courier_time'].fillna(X_features['dire_flying_courier_time'].mean(), inplace=True)
# X_features['dire_first_ward_time'].fillna(X_features['dire_first_ward_time'].mean(), inplace=True)

crossvalidator = KFold(n_splits=4, shuffle=True)
for n in [700]:
    start_time = datetime.datetime.now()
    # При увеличении количества деревьев время обучения растет быстрее
    # learning rate 0.1 маловато вроде
    # min_samples_split - При очень больших значениях ухудшает результат, значений для улучшения не нашел
    # max_features - при уменьшении увеличивет скорость обучения, рекомендуют искать возле корня от признаков
    # max_depth - при уменьшении увеличивaет скорость обучения
    # max_leaf_nodes - похоже на max_depth
    # subsamle - подвыборка для дерева, рекомендуют искать около 0.8
    clf = GradientBoostingClassifier(n_estimators=n, learning_rate=0.2, max_depth=2, max_features=4)
    crossvalidator.split(X=X_features)
    # auc - площадь под ROC кривой
    auc = cross_val_score(clf, X_features, Y_features['radiant_win'], cv=crossvalidator, scoring='roc_auc',
                          n_jobs=4)
    print(f"n_estimators={n}", "auc =", sum(auc)/len(auc))
    print("Duration:", datetime.datetime.now() - start_time)

#####################ОТЧЕТ#############################################
# 1. В 12 столбцах из 102 значение меньше, чем количество строк. Это значит что событие (покупка или установка
# предмета, первая кровь) не успело произойти в течении первых 5 минут, как это и указано в описании признаков
# Возможно у команд была стратегия выжидания. Признаки с пропусками:
# first_blood_time, first_blood_team, first_blood_player1, first_blood_player2,
# radiant_bottle_time, radiant_courier_time, radiant_flying_courier_time, radiant_first_ward_time
# dire_bottle_time, dire_courier_time, dire_flying_courier_time, dire_first_ward_time
# 2. Столбец с целевой переменной - radiant_win. Но можно уйти от задачи классификации к задаче регресcии,
# если сделать признак из столбцов tower_status
# 3. Кросс-валидация на 30 деревьях заняла 55с. AUC=0.69
# 4. Увеличение количества деревьев может немного улучшить обучение. Если ограничить глубину дерева и количество
# признаков, то можно ускорить обучение. С параметрами n_estimators=700, learning_rate=0.2, max_depth=2,
# max_features=4 разница 3% - AUC=0.72. Время 52с.