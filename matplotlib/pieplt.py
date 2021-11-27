import matplotlib.pyplot as plt
import numpy as np

data = [30, 25, 20, 12, 10]  # Это не проценты а соотношения
plt.figure(num=1, figsize=(6, 6))  # num это номер фигуры, ссылка на этот пирог
plt.axes(aspect=1)  # Значение должно растянивать график в одном из направлений, но почему-то не работает
plt.title('Plot 3', size=14)
plt.pie(data, labels=('Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5'))
plt.show()
