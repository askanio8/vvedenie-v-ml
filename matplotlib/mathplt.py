import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, LinearLocator


def pymath(x):
    a = (1.0 / (math.e * math.sin(x) + 1)) / (5.0 / 4.0 + 1.0 / x ** 15)
    res = math.log(a, math.e) / math.log(1 + x**15)  # Формула перехода
    return res


def npmath(x):
    a = np.log((1.0 / (math.e * math.sin(x) + 1)) / (5.0 / 4.0 + 1.0 / x ** 15))
    res = a / np.log(1 + x ** 15)  # Формула перехода
    return res


#x = float(input())
#npmath(x)  # nan
#pymath(x)  # Логарифм вычисляется не при всех действительных x exception


x = np.arange(-10, 10.01, 0.3)
plt.figure(figsize=(14, 15))  # размеры графика мб
plt.plot(x, np.sin(x), 'r--o',)  # Красные черточки
plt.plot(x, np.cos(x), 'bs', label=r'$f_2(x)=\cos(x)$')  # Синие  квадраты
plt.plot(x, -x, 'g^', label=r'$f_3(x)=-x$')  # Зеленые треугольники

plt.fill_between(x, np.sin(x), np.cos(x), where=x>0)  # закраска между графиками
plt.fill_between(x, np.sin(x), np.cos(x), where=x<0, color='g', alpha=0.5)

plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
#plt.title(r'$f_1(x)=\sin(x),\ f_2(x)=\cos(x),\ f_3(x)=-x$')
plt.grid(True)  # сетка
plt.axis([-10, 10, -10, 10])  # границы сетки
plt.xlim(-10, 10)  # или так
plt.ylim(-10, 10)
plt.legend(loc='best',fontsize=12)
#plt.savefig('figure_with_legend.png') Сохранение в файл
plt.show()
