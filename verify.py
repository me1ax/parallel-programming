import numpy as np
import matplotlib.pyplot as plt
import os

# Меняем директорию на папку с файлом скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Данные для графиков
sizes = []
times_1 = []
times_2 = []
times_4 = []

# Читаем результаты из файла all_results.txt
with open("all_results.txt", 'r') as f:
    lines = f.readlines()
    
# Пропускаем заголовок
for line in lines[1:]:
    parts = line.strip().split(',')
    if len(parts) == 4:
        try:
            size = int(parts[0])
            procs = int(parts[1])
            time_val = float(parts[2])
            
            if size not in sizes:
                sizes.append(size)
            
            if procs == 1:
                times_1.append(time_val)
            elif procs == 2:
                times_2.append(time_val)
            elif procs == 4:
                times_4.append(time_val)
        except:
            pass

# Сортируем по размеру матрицы
sorted_indices = sorted(range(len(sizes)), key=lambda i: sizes[i])
sizes = [sizes[i] for i in sorted_indices]
times_1 = [times_1[i] for i in sorted_indices]
times_2 = [times_2[i] for i in sorted_indices]
times_4 = [times_4[i] for i in sorted_indices]

print(f"Sizes: {sizes}")
print(f"Times 1 process: {times_1}")
print(f"Times 2 processes: {times_2}")
print(f"Times 4 processes: {times_4}")

# Проверка корректности умножения для матрицы 200x200 (или любой другой)
print("\nVerification for 200x200:")
try:
    # Читаем размер из файла
    with open("matrix_a.txt", 'r') as f:
        n = int(f.readline().strip())
    
    # Пропускаем первую строку с размером
    A = np.loadtxt("matrix_a.txt", skiprows=1)
    B = np.loadtxt("matrix_b.txt", skiprows=1)
    C_correct = np.dot(A, B)
    
    # Проверяем результаты для разных количеств процессов
    for p in [1, 2, 4]:
        fname = f"result_matrix_{n}_{p}p.txt"
        if os.path.exists(fname):
            C_res = np.loadtxt(fname, skiprows=1)
            if np.allclose(C_res, C_correct, rtol=1e-5):
                print(f"  {p} processes: OK")
            else:
                print(f"  {p} processes: FAIL")
        else:
            print(f"  {p} processes: file not found")
except Exception as e:
    print(f"Verification skipped: {e}")

# Построение графика
plt.figure(figsize=(10, 6))

plt.plot(sizes, times_1, 'bo-', label='1 process', linewidth=2, markersize=8)
plt.plot(sizes, times_2, 'go-', label='2 processes', linewidth=2, markersize=8)
plt.plot(sizes, times_4, 'ro-', label='4 processes', linewidth=2, markersize=8)

plt.xlabel('Matrix size N', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Matrix multiplication time vs size (MPI)', fontsize=14)
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
print(f"{'N':<8} {'1 process':<12} {'2 processes':<12} {'4 processes':<12} {'Speedup 4x':<12}")
print("-"*60)

for i in range(len(sizes)):
    if i < len(times_1) and i < len(times_4):
        speedup = times_1[i] / times_4[i] if times_4[i] > 0 else 0
        print(f"{sizes[i]:<8} {times_1[i]:<12.4f} {times_2[i]:<12.4f} {times_4[i]:<12.4f} {speedup:<12.2f}x")

print("\nГрафик сохранен как graph.png")