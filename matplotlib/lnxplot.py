import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P


x = np.arange(-20, 20.01, 0.01)

#x = np.delete(x, 200)  # если х в знаменателе убираем 0

plt.figure(figsize=(14, 15))  # размеры графика мб
plt.plot(x, np.log(x), 'r--', label=r'$f_1(x)=\ln(x)$')  # Красные черточки

plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')

plt.grid(True)
plt.legend(loc='best', fontsize=12)

plt.show()
