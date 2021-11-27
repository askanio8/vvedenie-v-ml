from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


ax = axes3d.Axes3D(plt.figure())
i = np.arange(-1, 1, 0.01)
X, Y = np.meshgrid(i, i)
Z = X**2 - Y**2
print(X)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)  # рисует 1 из 10 точек в каждом измерении
plt.show()