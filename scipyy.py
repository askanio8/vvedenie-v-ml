import numpy as np
from scipy import optimize
from scipy import linalg

# поиск мимнимума функции
a = optimize.minimize(lambda x: x ** x, [5])
print(a)

# решение СЛАУ для квадратных матриц
a = np.array([[5, 2, 7], [8, 5, 3], [1, 0, 1]])
b = np.array([7, 7, 1])
x = linalg.solve(a, b)
print(x)

mul = np.dot(a, x)  # матричное произведение
print(mul)

det = np.linalg.det(a)  # определитель
print(det)

rank = np.linalg.matrix_rank(a)  # ранг матрицы
print('rank:', rank)

inv = np.linalg.pinv(a)  # обратная матрица, если det=0
pinv = np.linalg.inv(a)  # обратная матрица, работает быстрее чем метод numpy
print('обратная:', inv)

# приближенное решение СЛАУ для неквадратных матриц
a = np.array([[5, 2, 7, 1], [8, 5, 3, 1], [1, 0, 1, 6]])
b = np.array([7, 7, 1])
x = linalg.lstsq(a, b)
print(x)
mul = np.dot(a, x[0])  # матричное произведение
print(mul)  # получается!

n = linalg.norm(b, ord=1)  # длина вектора манхэттенское расстояние
n = linalg.norm(b, ord=2)  # длина вектора евклидово расстояние
n = linalg.norm(b, ord=3)  # длина вектора 3ст
print(n)


import scipy.integrate as integrate
def f(x):
    return x**2
result = integrate.quad(f, 0, 1)  # вычисление интеграла ф-ии f в диапазоне [0, 1] (площадь под графиком)
print(result)

from scipy.misc import derivative
def f(x):
    return x**3 + x**10  # f' = 3x**2 + 10x**9 = 13(в точке 1)
result = derivative(f, 1.0, dx=1e-6)  # производная ф-ии f в точке 1.0
print(result)

from scipy.optimize import minimize
def func(x):
    return (x-1)**2 + 4
# x0 - стартовая точка, nelder-mead - один из методов поиска минимума ф-ии
res = minimize(func, x0=10000, method='nelder-mead', options={'xatol' : 1e-8, 'disp' : True})
print(res)

# Интерполяция ф-ии по точкам
from scipy.interpolate import interp1d
x = np.linspace(0, 10, num=11, endpoint=True)  # 11 точек в диапазоне от 0 до 10
y = np.cos(x)
f = interp1d(x, y)  # кусочно линейная интерполяция
f2 = interp1d(x, y, kind='cubic')  # кусочно кубическая интерполяция
xnew = np.linspace(0, 10, num=41, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()
