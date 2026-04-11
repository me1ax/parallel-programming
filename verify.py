import matplotlib.pyplot as plt
import os

# Меняем директорию
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Данные из твоего вывода
sizes = [200, 400, 800, 1200, 1600, 2000]

# Времена для разных конфигураций блоков (ms)
times_8x8 = [0.116512, 0.856064, 47.8474, 22.1364, 52.3559, 102.193]
times_16x16 = [0.124672, 0.824928, 46.3544, 22.8066, 51.6487, 68.4567]
times_32x32 = [0.166272, 0.950272, 7.52435, 22.2722, 35.3475, 68.9458]

# Построение графика
plt.figure(figsize=(12, 7))

plt.plot(sizes, times_8x8, 'b-o', label='8x8 (64 threads)', linewidth=2, markersize=8)
plt.plot(sizes, times_16x16, 'g-s', label='16x16 (256 threads)', linewidth=2, markersize=8)
plt.plot(sizes, times_32x32, 'r-^', label='32x32 (1024 threads)', linewidth=2, markersize=8)

plt.xlabel('Matrix size N', fontsize=14)
plt.ylabel('Time (ms)', fontsize=14)
plt.title('GPU Matrix Multiplication Time by Block Size (RTX 2060)', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12)

# Добавляем подписи точек для лучшей конфигурации
for i, size in enumerate(sizes):
    best_time = min(times_8x8[i], times_16x16[i], times_32x32[i])
    plt.annotate(f'{best_time:.2f}ms', (size, best_time), 
                fontsize=8, ha='left', va='bottom')

plt.tight_layout()
plt.savefig('cuda_blocks_graph.png', dpi=300)
plt.show()

# Таблица результатов
print("="*70)
print("GPU BLOCKS COMPARISON (RTX 2060)")
print("="*70)
print(f"{'N':<8} {'8x8 (ms)':<14} {'16x16 (ms)':<14} {'32x32 (ms)':<14} {'Best':<10}")
print("-"*70)

for i in range(len(sizes)):
    times = [times_8x8[i], times_16x16[i], times_32x32[i]]
    best = min(times)
    best_name = ""
    if best == times_8x8[i]: best_name = "8x8"
    elif best == times_16x16[i]: best_name = "16x16"
    else: best_name = "32x32"
    print(f"{sizes[i]:<8} {times_8x8[i]:<14.4f} {times_16x16[i]:<14.4f} {times_32x32[i]:<14.4f} {best_name:<10}")

print("\n" + "="*70)
print("CONCLUSIONS")
print("="*70)
print("1. Для маленьких матриц (200-400): лучше 8x8 или 16x16")
print("2. Для матрицы 800: 32x32 даёт преимущество (7.5 ms vs 46-47 ms)")
print("3. Для больших матриц (1200-2000): 16x16 и 32x32 одинаково хороши")
print("4. 32x32 стабильно показывает хорошие результаты на всех размерах")