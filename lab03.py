# ЛР3 - Реализация цифровой свёртки
# I Типовые сигналы: прямоугольный имульс (ПИ) и сигнал Гаусса (СГ)
# II Функция импульсного отклика ПИ и СГ
# III  Свёртки: ПИ + ПИ, ПИ + СГ, СГ + СГ, графики

from math import exp, pi
import numpy as np
import matplotlib.pyplot as plt
from time import time

T = 5
sigma = 1

n = 100
t_values = np.linspace(-2*T, 2*T, n)
t_values_2 = np.linspace(-2*T, 2*T, 2*n)

def rect_func(t):
    return 1 if -T < t and t < T else 0

def gauss_func(t):
    return exp(-t**2 / sigma**2)

def get_discrete_values(func):
    values = []
    for x in t_values:
        values.append(func(x))
    l = len(t_values)
    for x in range(l):
        values.append(0)
    return values

def norma(lst):
    return [x.real ** 2 + x.imag ** 2 for x in lst]

rect = np.array(get_discrete_values(rect_func))
gauss = get_discrete_values(gauss_func)

rect_f = np.fft.fft(rect)
gauss_f = np.fft.fft(gauss)

rect_m = rect_f * rect_f
# plt.plot(t_values_2, rect_m, 'r')
# plt.legend(loc="upper left")
# plt.show()

rect_rect = np.fft.ifft(rect_m, len(t_values_2))
rect_gauss = np.fft.ifft([rect_f[i] * gauss_f[i] for i in range(len(rect_f))], len(t_values_2))
gauss_gauss = np.fft.ifft([gauss_f[i] * gauss_f[i] for i in range(len(rect_f))], len(t_values_2))

# rect_rect = norma(rect_rect)
# rect_gauss = norma(rect_gauss)
# gauss_gauss = norma(gauss_gauss)

# plt.plot(t_values_2, rect, 'r', label='ПИ + ПИ')
# plt.legend(loc="upper left")
# plt.show()

plt.plot(t_values_2, rect_rect[:len(t_values_2)], 'r', label='ПИ + ПИ')
plt.legend(loc="upper left")
plt.show()
plt.plot(t_values_2, rect_gauss[:len(t_values_2)], 'r', label='ПИ + СГ')
plt.legend(loc="upper left")
plt.show()
plt.plot(t_values_2, gauss_gauss[:len(t_values_2)], 'r', label='СГ + СГ')
plt.legend(loc="upper left")
plt.show()