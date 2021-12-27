from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


ax = axes3d.Axes3D(plt.figure())
i = np.arange(-1, 1, 0.01)
X, Y = np.meshgrid(i, i)
Z = X**2 - Y**2
print(X)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)  # рисует 1 из 10 точек в каждом измерении
#ax.contour(X, Y, Z, rstride=10, cstride=10)  # линии уровня
#ax.contourf(X, Y, Z, rstride=10, cstride=10)  # поверхности уровня
plt.show()


fig, ax = plt.subplots(1, 2)
i = np.arange(-1, 1, 0.01)
X, Y = np.meshgrid(i, i) # создание двумерных массивов. если есть X Y Z одномерные, то можно использовать функции
# triccontour и triccontourf
Z = X**2 - Y**2
print(X)
c1 = ax[0].contour(X, Y, Z, 15)  # линии уровня
c1.clabel()  # значения глубины на линиях
ax[1].contourf(X, Y, Z)  # поверхности уровня
plt.show()