import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt


def makeData():
    x = numpy.arange(-10, 10, 0.1)
    y = numpy.arange(-10, 10, 0.1)
    xgrid, ygrid = numpy.meshgrid(x, y)  # возвращает 2 экз матриц поверхности в 2 ракурсах, принимает 2 вектора
    # a = xgrid[63][11] == ygrid[11][63]
    zgrid = numpy.sin(xgrid) * numpy.sin(ygrid) / (xgrid * ygrid)
    return xgrid, ygrid, zgrid


x, y, z = makeData()

fig = pylab.figure()  # pylab вместо plt можно вращать и масштабировать график
axes = Axes3D(fig)
axes.plot_surface(x, y, z, rstride=4, cstride=4, cmap='jet')  # color maps тут можно выбрать цвета

#plt.show()
pylab.show()
