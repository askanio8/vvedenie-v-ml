# Stacking - Результаты множества плохих моделей(возможно разнотипных) передаются на входы другой модели вместе
# с исходными признаками

# Bagging(собственно random forest) - Результаты множества плохих моделей одного типа усредняются, прогноз должен
# быть лучше. Для того, чтобы построенные деревья отличались друг от друга, каждое из них обучают на подмножестве
# признаков подмножества объектов обучающей выборки (в случае нейросетей также не достаточно перемешать объекты и
# использовать другое случайное зерно из за проблемы декореляции - если подвыборки очень большие, то построенные
# модели будут слишком похожи. Это и есть проблема слишком больших сетей - много одинаково бученных нейронов
# дублируют друг друга)
# Информативность каждого признака измеряется путем премешивания значений этого признака в обучающей выборке.
# Затем обучающая выборка подаётся в модель и сравнивается результат с нормальной выборкой. Неинформативные
# признаки потом можно выбросить, возможно модель улучшится.

# Boosting - взвешенный бэггинг.
# Градиентный бустинг - оценка ошибок y-y0 плохих моделей передаются следующей модели, во время обучения модель
# получает сам объект и ошибку на нём предыдущих моделей. У каждой модели есть вес α.Количество моделей подбирается
# с помощью требуемой ошибки или её градиента
# Adaboost - каждому объекту обучающей выборки назначается вес 1/l, Затем строитя простая модель, вычисляется
# взвешенная по весам объектов ошибка обучения. В зависимости от неё по простой эмпиричесой формуле вычисляем вес
# этой модели. Пересчитываем веса объектов, чем больше была ошибка на объекте, тем больший вес он получает. Норм
# ируем веса объектов, чтобы в сумме было 1. Строим следующую модель, она будет больше внимания обрашать на объекты
# с большим весом... Должен быть критерий останова. Adaboost чувствителен к выбросам, модели после 20 стоит
# проверить веса объектов, объекты с большими весами возможно выбросы. Adaboost можно использовать в качестве
# детектора выбросов во время подготовки обучающей выборки к другим алгоритмам.
# Xgboost - градиентный бустинг с регуляризацией по весам листьев деревьев и по их количеству
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score  # одна из метрик качества, в cross_val_score есть параметр r2
from sklearn.model_selection import cross_val_score


# В этой задаче нужно предсказывать возраст ракушки по её измерениям
data = pd.read_csv('abalone.csv')
#  Преобразуем столбец в числовые значения
data['Sex'] = data['Sex'].map(lambda x: 1 if x=='M' else (-1 if x=='F' else 0))

Xdata = data.drop(columns='Rings')
Ydata = data.filter(items=['Rings'])

r2 = [] # Список оценок качества обучения леса
crossvalidator = KFold(shuffle=True, random_state=1, n_splits=5)
for n in range(1, 51):
    # для случайного леса важен подбор параметров обучения - количество деревьев, максимальная
    # глубина дерева, загрязненность узла и тд
    clf = RandomForestRegressor(random_state=1, n_estimators=n)
    randnumbers = crossvalidator.split(Xdata)

    res = cross_val_score(clf, Xdata, Ydata.values.ravel(), cv=crossvalidator, scoring='r2')
    # Вложенный цикл можно было не городить, а использовать cross_val_score с параметром scoring='r2', как это
    # сделано строкой выше
    #res = []
    #for train, test in randnumbers:
    #    clf.fit(Xdata.iloc[train], Ydata.iloc[train].values.ravel())
    #    result = clf.predict(Xdata.iloc[test])
    #    res.append(r2_score(Ydata.iloc[test].values.ravel(), result))
    #r2.append(sum(res)/len(res))
    print(n, ':', sum(res)/len(res))

# 28 деревьев нужно для качества выше 0.525