import  numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(7,4), facecolor='lightblue')
ax = fig.add_subplot(facecolor='lightgreen')

# всем надписям можно установить шрифт, цвет, прозрачность, поворот, выравнивание и тд
# еще параметр bbox определяет стиль рамки текста
plt.figtext(0.05, 0.6, "Текст в области окна")
fig.suptitle("Заголовок окна")
ax.set_xlabel("0x")
ax.set_ylabel("0y")
ax.text(0.05, 0.1, "Текст в координатных осях")
ax.annotate("Аннотация", xy=(0.2, 0.4), xytext=(0.6, 0.7),
            arrowprops={"facecolor": "gray", "shrink": 0.1})

# ниже запятая нужна
line1, = ax.plot(np.arange(0, 5, 0.25), '--o', label='line1')
line2, = ax.plot(np.arange(0, 10, 0.5), ':s', label='line2')

ax.legend((line1, line2), [r'$f(x) = a \cdot b + c$', r"$f = LaTeX$ $нотация$"])


plt.show()