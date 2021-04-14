import numpy as np
import time
import matplotlib.pyplot as plt

X = np.random.normal(loc=1, scale=10, size=(1000, 50))

t = time.time()
bars = []
for i in range(50):
    #bars.append(len([True for n in X.flat if -25.0 + i < n < -24.0 + i]))  # Строка ниже делает то же самое
    bars.append(len(np.nonzero(np.logical_and(X > -25.0 + i, X < -24.0 + i))[0]))  # Но в 30 раз быстрее

#print(time.time() - t, bars)

y_pos = np.arange(len(bars))  # Рисуем график
plt.bar(y_pos, bars, align='center', alpha=0.5)
plt.show()

m = np.mean(X, axis=0)  # Среднее значение по столбцам
std = np.std(X, axis=0)  # Среднеквадратичное отклонение по столбцам(примерно scale=10)
X_norm = ((X - m) / std)  # Нормировка матрицы по столбцам
# print(X_norm)
##################################################################################################
Z = np.array([[4, 5, 0],
              [1, 9, 3],
              [5, 1, 1],
              [3, 3, 3],
              [9, 9, 9],
              [4, 7, 1]])

# sums = [sum(x) for x in Z]
# for i, x in enumerate(sums):
#    if x > 10:
#        print(i)

r = np.sum(Z, axis=1)  # Так лучше
print(np.nonzero(r > 10)[0])  # Номера строк, сумма в которых больше 10
###########################################################################################################
a = np.eye(3,3)
b = np.eye(3,3)
c = np.vstack((a, b))  # Вертикальное объединение матриц
print(c)