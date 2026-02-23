import numpy as np
import matplotlib.pyplot as plt

# Загрузка выборки

filename = 'variant_13.csv' 
data = np.loadtxt(filename, delimiter=',')

n = len(data)
print(f"Объём выборки: n = {n}")
print()


# Вариационный ряд

sorted_data = np.sort(data)

print("Вариационный ряд (фрагмент):")
print(f"  Первые 5 значений: {sorted_data[:5]}")
print(f"  Последние 5 значений: {sorted_data[-5:]}")
print()


# Выборочные оценки
x_bar = np.mean(data)               # выборочное среднее
s2 = np.var(data, ddof=1)           # несмещенная дисперсия 
s = np.sqrt(s2)                     # стандартное отклонение (исправленное)
median = np.median(data)            # медиана
x_min = np.min(data)                # минимум
x_max = np.max(data)                # максимум

print("Выборочные оценки:")
print(f"  Среднее:      x̄ = {x_bar:.4f}")
print(f"  Дисперсия:    s² = {s2:.4f}")
print(f"  Ст. откл.:    s = {s:.4f}")
print(f"  Медиана:      x̃ = {median:.4f}")
print(f"  Размах:       [{x_min:.2f}, {x_max:.2f}] (R = {x_max - x_min:.2f})")
print()


# Правило Скотта и гистограмма

# h — оптимальная ширина интервала по Скотту
h = 3.5 * s * n**(-1/3)
# k — количество интервалов
k = int(np.ceil((x_max - x_min) / h))

print(f"Правило Скотта:")
print(f"  Ширина интервала: h = {h:.2f}")
print(f"  Число интервалов: k = {k}")
print()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Гистограмма по правилу Скотта
ax1.hist(data, bins=k, edgecolor='black', color='skyblue', alpha=0.7, density=True)
ax1.axvline(x_bar, color='red', linestyle='--', label=f'x̄ = {x_bar:.2f}')
ax1.set_title(f"Гистограмма (Скотт, k={k})")
ax1.legend()

# Гистограмма с 5 интервалами
ax2.hist(data, bins=5, edgecolor='black', color='lightgreen', alpha=0.7, density=True)
ax2.axvline(x_bar, color='red', linestyle='--', label=f'x̄ = {x_bar:.2f}')
ax2.set_title("Гистограмма (k=5)")
ax2.legend()

plt.show()


# Полигон частот
plt.figure(figsize=(8, 5))

counts, bins = np.histogram(data, bins=k, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

plt.plot(bin_centers, counts, marker='o', linestyle='-', color='blue')
plt.fill_between(bin_centers, counts, alpha=0.1, color='blue')
plt.title("Полигон частот (плотность)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlabel("Значение")
plt.ylabel("Плотность")
plt.show()


# Эмпирическая функция распределения (ЭФР)

plt.figure(figsize=(8, 5))

x_ecdf = sorted_data
y_ecdf = np.arange(1, n + 1) / n

plt.step(x_ecdf, y_ecdf, where='post', color='darkmagenta', label='ЭФР')

plt.scatter(x_ecdf, y_ecdf, color='darkmagenta', s=10)
for x, y in zip(x_ecdf[::5], y_ecdf[::5]): 
    plt.vlines(x, y - 1/n, y, colors='gray', linestyles='dashed', alpha=0.5)

plt.title("Эмпирическая функция распределения")
plt.xlabel("x")
plt.ylabel("F*(x)")
plt.grid(True, alpha=0.3)
plt.show()


# Сравнение с истинными параметрами
mu_true = 110  
sigma2_true = 144 

print("Сравнение с истинными параметрами:")
print(f"  Истинное μ = {mu_true}, выборочное x̄ = {x_bar:.4f}")
print(f"  Истинное σ² = {sigma2_true}, выборочное s² = {s2:.4f}")
print()
print("Ответ на вопрос:")
print("Выборочные оценки отличаются от истинных параметров из-за случайной природы выборки.")
print("Поскольку мы работаем с ограниченным набором данных (n=100), возникает ошибка репрезентативности.")
print("Согласно закону больших чисел, при увеличении n выборочное среднее будет стремиться к истинному.")