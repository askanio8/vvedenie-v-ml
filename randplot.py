import numpy as np
import matplotlib.pyplot as plt

# Генерируем случайные наборы данных для графика
np.random.seed(0)  # Случайное зерно для воспроизводимости результата
l = 100  # Количество строк (точек на графике)
n = 2  # Количество столбцов (размероность графика - тут 2 координаты, обе случайные)

# 2 значения матем. ожидания + среднеквадратичное отклонение * двумерный массив нормально распределенных значений
X1 = np.array([[-1, -1]]) + 0.5 * np.random.randn(l, n)
X2 = np.array([[1, 1]]) + 0.5 * np.random.randn(l, n)
X3 = np.array([[-1, 1]]) + 0.5 * np.random.randn(l, n)

#  Рисуем график
X = [X1, X2, X3]
y = [0, 1, 2]  # Классы
cols = ["blue", "red", "green"]
for k in y:
    plt.plot(X[k][:, 0], X[k][:, 1], "o", label="класс {}".format(k), color=cols[k])
plt.legend(loc='best')
plt.show()
