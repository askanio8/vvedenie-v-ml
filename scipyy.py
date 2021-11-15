import numpy as np
from scipy import optimize
from scipy import linalg


# поиск мимнимума функции
a = optimize.minimize(lambda x: x**x, [5])
print(a)

# решение СЛАУ для квадратных матриц
a = np.array([[5, 2, 7], [8, 5, 3], [1, 0, 1]])
b = np.array([7, 7, 1])
x = linalg.solve(a, b)
print(x)

mul = np.dot(a,x)  # матричное произведение
print(mul)

det = np.linalg.det(a)  # определитель
print(det)

rank = np.linalg.matrix_rank(a)  # ранг матрицы
print('rank:', rank)

inv = np.linalg.pinv(a)  # обратная матрица, если det=0
pinv = np.linalg.inv(a)  # обратная матрица
print('обратная:', inv)


# приближенное решение СЛАУ для неквадратных матриц
a = np.array([[5, 2, 7, 1], [8, 5, 3, 1], [1, 0, 1, 6]])
b = np.array([7, 7, 1])
x = linalg.lstsq(a, b)
print(x)
mul = np.dot(a,x[0])  # матричное произведение
print(mul)  # получается!


n = linalg.norm(b, ord=1)  # длина вектора манхэттенское расстояние
n = linalg.norm(b, ord=2)  # длина вектора евклидово расстояние
n = linalg.norm(b, ord=3)  # длина вектора 3ст
print(n)

