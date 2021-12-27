import numpy as np
import matplotlib.pyplot as plt

# ступеньки
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot()
x = np.arange(0, 10)
ax.step(x, x, '-go', x, np.cos(x), '--x', where='mid')
ax.grid()
plt.show()

# стек
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot()
x = np.arange(-2, 2, 0.1)
y1 = np.array([-y ** 2 for y in x]) + 8
y2 = np.array([-y ** 2 for y in x]) + 8
y3 = np.array([-y ** 2 for y in x]) + 8
ax.stackplot(x, y1, y2, y3)
ax.grid()
plt.show()

# поплавки над линией
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot()
x = np.arange(-np.pi, np.pi, 0.3)
ax.stem(x, np.cos(x), '--r', '^g', bottom=0.5)
ax.grid()
plt.show()

# облако точек
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot()
x = np.random.normal(0, 2, 500)
y = np.random.normal(0, 2, 500)
ax.scatter(x, y, s=50, c='g', linewidths=1, marker='s', edgecolors='r')
ax.grid()
plt.show()

# гистограмма
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot()
y = np.random.normal(0, 2, 500)
ax.hist(y, 50)  # среднее по столбцам считает сама
ax.grid()
plt.show()

# просто столбчатая
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot()
x = [f'H{i + 1}' for i in range(10)]
y = np.random.randint(1, 5, len(x))
ax.bar(x, y)
# ax.barh(x, y)  # столбцы по горизонтали
ax.grid()
plt.show()

# сезонная столбчатая
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot()
x = np.arange(10)
y1 = np.random.randint(3, 20, len(x))
y2 = np.random.randint(3, 20, len(x))
w = 0.3
ax.bar(x-w/2, y1, width=w)
ax.bar(x+w/2, y2, width=w)
ax.grid()
plt.show()

# круговая
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot()
vals = [10, 40 ,23, 30, 7]
labels = ['Toyota', 'BMW', 'Lexus', 'Audi', 'Lada']
exp = [0.1, 0, 0.3, 0, 0]
ax.pie(vals, labels=labels, autopct='%0.2f', explode=exp, shadow=True, wedgeprops=dict(width=0.5))
ax.grid()
plt.show()