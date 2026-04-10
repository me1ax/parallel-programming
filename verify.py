import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

sizes = []
times_1 = []
times_2 = []
times_4 = []

with open("experiment_results.txt", 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            size = int(parts[0])
            threads = int(parts[1])
            time_val = float(parts[2])
            
            if size not in sizes:
                sizes.append(size)
            
            if threads == 1:
                times_1.append(time_val)
            elif threads == 2:
                times_2.append(time_val)
            elif threads == 4:
                times_4.append(time_val)

# Сортируем по размеру
sorted_indices = sorted(range(len(sizes)), key=lambda i: sizes[i])
sizes = [sizes[i] for i in sorted_indices]
times_1 = [times_1[i] for i in sorted_indices]
times_2 = [times_2[i] for i in sorted_indices]
times_4 = [times_4[i] for i in sorted_indices]

print(f"Sizes: {sizes}")
print(f"Times 1 thread: {times_1}")
print(f"Times 2 threads: {times_2}")
print(f"Times 4 threads: {times_4}")

# Проверка корректности умножения для матрицы 2000x2000
print("\nVerification for 2000x2000:")
try:
    with open("a.txt", 'r') as f:
        n = int(f.readline().strip())
    
    A = np.loadtxt("a.txt", skiprows=1)
    B = np.loadtxt("b.txt", skiprows=1)
    C_correct = np.dot(A, B)
    
    for t in [1, 2, 4]:
        fname = f"result_{t}.txt"
        if os.path.exists(fname):
            C_res = np.loadtxt(fname, skiprows=1)
            if np.allclose(C_res, C_correct, rtol=1e-5):
                print(f"  {t} threads: OK")
            else:
                print(f"  {t} threads: FAIL")
        else:
            print(f"  {t} threads: file not found")
except Exception as e:
    print(f"Error: {e}")

# Построение графика
plt.figure(figsize=(10, 6))

plt.plot(sizes, times_1, 'bo-', label='1 thread', linewidth=2, markersize=8)
plt.plot(sizes, times_2, 'go-', label='2 threads', linewidth=2, markersize=8)
plt.plot(sizes, times_4, 'ro-', label='4 threads', linewidth=2, markersize=8)

plt.xlabel('Matrix size N', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Matrix multiplication time vs size (OpenMP)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Добавляем подписи значений для больших матриц
for i, size in enumerate(sizes):
    if size >= 800:
        plt.annotate(f'{times_1[i]:.2f}s', (size, times_1[i]), fontsize=8, alpha=0.7, ha='left')
        plt.annotate(f'{times_4[i]:.2f}s', (size, times_4[i]), fontsize=8, alpha=0.7, ha='left')

plt.tight_layout()
plt.savefig('graph.png', dpi=300)
plt.show()

# Вывод таблицы результатов
print("\n" + "="*60)
print("RESULTS TABLE")
print("="*60)
print(f"{'N':<8} {'1 thread':<12} {'2 threads':<12} {'4 threads':<12} {'Speedup 4x':<12}")
print("-"*60)

for i in range(len(sizes)):
    speedup = times_1[i] / times_4[i] if times_4[i] > 0 else 0
    print(f"{sizes[i]:<8} {times_1[i]:<12.4f} {times_2[i]:<12.4f} {times_4[i]:<12.4f} {speedup:<12.2f}x")

print("\nГрафик сохранен как graph.png")