#include <iostream>
#include <vector>
#include <fstream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <iomanip>

using namespace std;

// Текстовая запись матрицы
void writeMatrixText(const string& filename, const vector<double>& M, int n) {
    ofstream file(filename);
    if (!file) {
        cerr << "Cannot create file: " << filename << endl;
        return;
    }
    
    file << n << endl;
    file << fixed << setprecision(6);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << M[i * n + j];
            if (j < n - 1) file << " ";
        }
        file << endl;
    }
    file.close();
    cout << "Result saved to: " << filename << " (text format)" << endl;
}

// Функция для проверки
void printSmallMatrix(const vector<double>& M, int n, int max_rows = 10) {
    cout << "Matrix preview (first " << min(max_rows, n) << "x" << min(max_rows, n) << "):" << endl;
    for (int i = 0; i < min(max_rows, n); i++) {
        for (int j = 0; j < min(max_rows, n); j++) {
            cout << fixed << setprecision(2) << M[i * n + j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 2) {
        if (rank == 0)
            cout << "Usage: " << argv[0] << " <matrix_size>" << endl;
        MPI_Finalize();
        return 1;
    }
    
    int n = atoi(argv[1]);
    
    vector<double> A, B;
    
    if (rank == 0) {
        srand(time(0));
        A.resize(n * n);
        B.resize(n * n);
        
        // Генерация маленьких чисел для читаемости
        for (int i = 0; i < n * n; i++) {
            A[i] = (rand() % 10) + 1;  // 1-10
            B[i] = (rand() % 10) + 1;
        }
        
        cout << "\n========================================" << endl;
        cout << "Matrix Multiplication: " << n << "x" << n << endl;
        cout << "Processes: " << size << endl;
        cout << "========================================\n" << endl;
    }
    
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        A.resize(n * n);
        B.resize(n * n);
    }
    
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    int rows_per_proc = n / size;
    int remainder = n % size;
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    vector<double> local_C(local_rows * n, 0.0);
    
    double start_time = MPI_Wtime();
    
    // Умножение матриц
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[(start_row + i) * n + k] * B[k * n + j];
            }
            local_C[i * n + j] = sum;
        }
    }
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        vector<double> C(n * n);
        
        // Копирование своей части
        copy(local_C.begin(), local_C.end(), C.begin() + start_row * n);
        
        // Сбор результатов от других процессов
        for (int p = 1; p < size; p++) {
            int p_start = p * rows_per_proc + min(p, remainder);
            int p_rows = rows_per_proc + (p < remainder ? 1 : 0);
            MPI_Recv(C.data() + p_start * n, p_rows * n, MPI_DOUBLE, 
                     p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Сохранение в ТЕКСТОВОМ формате
        writeMatrixText("result_matrix_" + to_string(n) + ".txt", C, n);
        
        // Показ маленькой части результата для проверки
        printSmallMatrix(C, n);
        
        // Вывод производительности
        double time = end_time - start_time;
        long long ops = 2LL * n * n * n;
        cout << "\n========================================" << endl;
        cout << "Time: " << time << " seconds" << endl;
        cout << "Performance: " << (ops / time / 1e9) << " GFLOPS" << endl;
        cout << "========================================" << endl;
        
    } else {
        MPI_Send(local_C.data(), local_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}