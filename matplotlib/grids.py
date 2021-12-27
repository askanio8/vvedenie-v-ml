import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, LinearLocator, \
    MultipleLocator, IndexLocator, FixedLocator, LogLocator, MaxNLocator
from matplotlib.ticker import NullFormatter, FormatStrFormatter

fig = plt.figure()
axes = fig.add_subplot (1, 1, 1)

y = np.random.random(10)
plt.plot(y)

plt.grid()  # сетка

plt.axis([-10, 10, -10, 10])  # границы сетки
plt.xlim(-10, 10)  # или так
plt.ylim(-10, 10)

axes.xaxis.set_major_locator(NullLocator())  # без линий по этой оси
axes.xaxis.set_major_locator(LinearLocator(7))  # 7 линий  по этой оси
axes.xaxis.set_major_locator(IndexLocator(base=3, offset=-1))  # сдвиг и первая линия
axes.xaxis.set_major_locator(FixedLocator([1, 3, 4, 8, 9]))  # линии по списку значений
axes.xaxis.set_major_locator(LogLocator(base=2))  # линии по списку значений
axes.xaxis.set_major_locator(MaxNLocator(8))  # линии с удобными значениями, но не больше параметра
axes.xaxis.set_major_locator(MultipleLocator(3))  # линии через 3 значеня

axes.minorticks_on()  # минорная сетка
axes.grid(which="major", lw=1, color='b', alpha=0.3)  # толщина линий 1
axes.grid(which="minor", lw= 0.5, color='b', alpha=0.3)
axes.xaxis.set_minor_locator(MultipleLocator(1/3))  # 7 линий  по этой оси

axes.yaxis.set_major_formatter(NullFormatter())  # не отображать значения делений сетки
axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))  # отображать значения 2 знака после запятой

plt.show()


# логарифмический масштаб
fig = plt.figure(figsize=(7,4))
x = np.arange(-10*np.pi, 10*np.pi, 0.1)
axes = fig.add_subplot ()
axes.plot(x, np.sinc(x) * np.exp(-np.abs(x/10)))
axes.set_yscale('log')
axes.grid()

plt.show()