# ЛР1 - Изучение дискретизации сингалов
# Типовые сигналы: прямоугольный имульс и сигнал Гаусса
# I Дискретизация при разных dt
# II Восстановление по теореме Котельникова
# III  Построение графиков: исходная и восстановленная функции
# IV Ответы на вопросы

from math import exp, sin, pi
import numpy as np
import matplotlib.pyplot as plt

T = 5
sigma = 1

n = 25
dt = 1
t_max = dt * (n - 1) / 2
t_values = np.arange(-t_max, t_max, dt)

step = 0.1

def rect_func(t):
    return 1 if -T < t and t < T else 0

def gauss_func(t):
    return exp(-t**2 / sigma**2)

def get_discrete_values(func):
    values = []
    for x in t_values:
        values.append(func(x))
    return values

def restore_function(discrete):
    xs = np.arange(-t_max, t_max, step)
    restored = [0] * len(xs)
    for i in range(len(xs)):
        for j in range(n - 1):
            restored[i] += discrete[j] * sin((xs[i] - t_values[j]) / dt * pi) / ((xs[i] - t_values[j]) / dt * pi)
    return restored, xs

rect = get_discrete_values(rect_func)
gauss = get_discrete_values(gauss_func)
restored_rect, xs = restore_function(rect)
restored_gauss, _ = restore_function(gauss)
plt.plot(t_values, rect, 'r', label='Исходный прямоугольный импульс')
plt.plot(xs, restored_rect, 'g', label='Восстановленный прямоугольный импульс')
plt.legend(loc="upper left")
plt.show()

plt.plot(t_values, gauss, 'r', label='Исходный сигнал Гаусса')
plt.plot(xs, restored_gauss, 'g', label='Восстановленный сигнал Гаусса')
plt.legend(loc="upper left")
plt.show()





