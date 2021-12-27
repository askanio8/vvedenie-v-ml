import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ws = [1, 2, 5]  # доли ширины для каждого столбца
hs = [2, 0.5]  # доли ширины для каждой строки

fig = plt.figure(figsize=(7, 4))
gs = GridSpec(ncols=3, nrows=2, figure=fig, width_ratios=ws, height_ratios=hs)

ax1 = plt.subplot(gs[0,0])
ax1.plot(np.arange(0, 5, 0.2))
ax2 = fig.add_subplot(gs[1, 0:2])
ax2.plot(np.random.random(10))
ax3 = fig.add_subplot(gs[:, 2])
ax3.plot(np.random.random(10))

plt.show()