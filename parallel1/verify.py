import numpy as np
import matplotlib.pyplot as plt
import os

# Переходим в папку, где лежит скрипт
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Чтение результатов из all_results.txt
sizes = []
times = []
operations = []

with open("all_results.txt", 'r') as f:
    next(f)
    for line in f:
        parts = line.strip().split(',')
        sizes.append(int(parts[0]))
        times.append(float(parts[1]))
        operations.append(int(parts[2]))

# Верификация
print("Верификация")
A = np.loadtxt("matrix_a.txt", skiprows=1)
B = np.loadtxt("matrix_b.txt", skiprows=1)
C_result = np.loadtxt("result_matrix.txt", skiprows=1)
C_correct = np.dot(A, B)

if np.allclose(C_result, C_correct, rtol=1e-5):
    print("PASSED")
else:
    print("FAILED")

# График
plt.figure(figsize=(8, 5))
plt.plot(sizes, times, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Размер матрицы N')
plt.ylabel('Время выполнения (с)')
plt.title('Зависимость времени от размера матрицы')
plt.grid(True, alpha=0.3)

for i, (size, time) in enumerate(zip(sizes, times)):
    plt.annotate(f'{time:.3f}', (size, time), xytext=(0,10), 
                textcoords="offset points", ha='center')

plt.savefig('graph.png', dpi=300, bbox_inches='tight')
plt.show()

# Таблица
print("\nТаблица")
print(f"{'N':<6} {'Операций':<15} {'Время (с)':<10}")
print("-" * 35)
for size, ops, time in zip(sizes, operations, times):
    print(f"{size:<6} {ops:<15.2e} {time:<10.4f}")