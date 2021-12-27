import time
import matplotlib.pyplot as plt
import numpy as np
# более простой способ использовать FuncAnimation, ArtistAnimation
from matplotlib.animation import FuncAnimation, ArtistAnimation

plt.ion()  # интерактивное отображение
fig, ax = plt.subplots()
x = np.arange(-2*np.pi, 2*np.pi, 0.1)
y = np.cos(x)
line, = ax.plot(x,y)

for delay in np.arange(0, np.pi, 0.1):
    y = np.cos(x +delay)

    line.set_ydata(y)
    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.01)

plt.ioff()
plt.show()