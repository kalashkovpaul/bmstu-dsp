# ЛР2 - Изучение БПФ (Быстрое преобразование Фурье)
# I Типовые сигналы: прямоугольный имульс и сигнал Гаусса
# II Выполнить ДПФ и БПФ
# III Определить время выполнения, N приблизительно 128,256

from math import exp, pi
import numpy as np
import matplotlib.pyplot as plt
from time import time

T = 5
sigma = 1

n = 128
# dt = 0.05
# t_max = dt * (n - 1) / 2
t_values = np.linspace(-2*T, 2*T, n)


def rect_func(t):
    return 1 if -T < t and t < T else 0

def gauss_func(t):
    return exp(-t**2 / sigma**2)

def get_discrete_values(func):
    values = []
    for x in t_values:
        values.append(func(x))
    return values

def norma(lst):
    return [x.real ** 2 + x.imag ** 2 for x in lst]

def dft(lst):
    n = len(lst)
    ks = np.arange(0, n, 1)
    r = np.zeros(n)
    i = 1.j
    for ki in range(n):
        for j in range(n):
            r[ki] += lst[j]*np.exp(-2*pi*ks[ki]*i * j / n)
    return r

rect = get_discrete_values(rect_func)
gauss = get_discrete_values(gauss_func)

start = time()
rect_fft = np.fft.fft(rect)
print(f'FFT, rect: {time() - start}')

start = time()
rect_dft = dft(rect)
print(f'DFT, rect: {time() - start}')

rect_shifted_dft = np.fft.fftshift(rect_dft)
rect_norma_twins_dft = np.abs(rect_dft)
rect_norma_dft = np.abs(rect_shifted_dft)
rect_shifted = np.fft.fftshift(rect_fft)
rect_norma_twins = np.abs(rect_fft)
rect_norma = np.abs(rect_shifted)
plt.plot(t_values, rect, 'r', label='Исходный прямоугольный импульс')
plt.legend(loc="upper left")
plt.show()
plt.plot(t_values, rect_norma_twins_dft, 'r', label='Прямоугольный импульс (DFT)')
plt.legend(loc="upper left")
plt.show()
plt.plot(t_values, rect_norma_dft, 'r', label='Прямоугольный импульс (DFT)')
plt.legend(loc="upper left")
plt.show()
plt.plot(t_values, rect_norma_twins, 'g', label='Прямоугольный импульс (FFT)')
plt.legend(loc="upper left")
plt.show()
plt.plot(t_values, rect_norma, 'b', label='Прямоугольный импульс (FFT)')
plt.legend(loc="upper left")
plt.show()

start = time()
gauss_fft = np.fft.fft(gauss)
print(f'FFT, Gauss: {time() - start}')

start = time()
gauss_dft = dft(gauss)
print(f'DFT, Gauss, {time() - start}')

gauss_shifted_dft = np.fft.fftshift(gauss_dft)
gauss_norma_twins_dft = np.abs(gauss_dft)
gauss_norma_dft = np.abs(gauss_shifted_dft)
gauss_shifted = np.fft.fftshift(gauss_fft)
gauss_norma_twins = np.abs(gauss_fft)
gauss_norma = np.abs(gauss_shifted)
plt.plot(t_values, gauss, 'r', label='Исходный сигнал Гаусса')
plt.legend(loc="upper left")
plt.show()
plt.plot(t_values, gauss_norma_twins_dft, 'r', label='Сигнал Гаусса (DFT)')
plt.legend(loc="upper left")
plt.show()
plt.plot(t_values, gauss_norma_dft, 'r', label='Сигнал Гаусса (DFT)')
plt.legend(loc="upper left")
plt.show()
plt.plot(t_values, gauss_norma_twins, 'g', label='Сигнал Гаусса (FFT)')
plt.legend(loc="upper left")
plt.show()
plt.plot(t_values, gauss_norma, 'b', label='Сигнал Гаусса (FFT)')
plt.legend(loc="upper left")
plt.show()



