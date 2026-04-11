#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

using namespace std;

// Ядро CUDA для умножения матриц
__global__ void matrixMulKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void generateMatrix(string filename, int n) {
    ofstream file(filename);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1.0, 10.0);

    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            file << dis(gen) << " ";
        file << endl;
    }
}

vector<vector<double>> readMatrix(string filename, int& n) {
    ifstream file(filename);
    file >> n;
    vector<vector<double>> matrix(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> matrix[i][j];
    return matrix;
}

vector<vector<double>> multiplySequential(vector<vector<double>> A, vector<vector<double>> B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

void writeResult(string filename, vector<vector<double>> C) {
    ofstream file(filename);
    int n = C.size();
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            file << C[i][j] << " ";
        file << endl;
    }
}

int main() {
    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int num_sizes = 6;

    // КОРРЕКТНЫЕ конфигурации блоков для RTX 2060 (макс 1024 потока)
    struct BlockConfig {
        int x;
        int y;
        const char* name;
    };

    BlockConfig blockConfigs[] = {
        {8, 8, "8x8 (64 threads)"},
        {16, 16, "16x16 (256 threads)"},
        {32, 32, "32x32 (1024 threads)"}
    };
    int numConfigs = 3;

    // Проверка CUDA устройств
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
        cerr << "Make sure CUDA Toolkit is installed and you have an NVIDIA GPU" << endl;
        return 1;
    }

    if (deviceCount == 0) {
        cerr << "No CUDA devices found!" << endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    cout << "========================================" << endl;
    cout << "CUDA Matrix Multiplication Benchmark" << endl;
    cout << "========================================" << endl;
    cout << "GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
    cout << "========================================" << endl;

    ofstream results("cuda_all_results.txt");
    results << "N,BlockSize,Threads,Time(ms),Speedup,Ops" << endl;
    results.close();

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];

        cout << "\nTesting N=" << n << "..." << endl;

        generateMatrix("matrix_a.txt", n);
        generateMatrix("matrix_b.txt", n);

        int n1, n2;
        auto A = readMatrix("matrix_a.txt", n1);
        auto B = readMatrix("matrix_b.txt", n2);

        // Последовательное умножение
        cout << "  Running sequential... ";
        auto start_seq = chrono::high_resolution_clock::now();
        auto C_seq = multiplySequential(A, B);
        auto end_seq = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> seq_time = end_seq - start_seq;
        cout << seq_time.count() << " ms" << endl;

        // Подготовка для CUDA
        int size = n * n;
        size_t bytes = size * sizeof(double);

        double* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        vector<double> A_flat(size), B_flat(size);
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                A_flat[row * n + col] = A[row][col];
                B_flat[row * n + col] = B[row][col];
            }
        }

        cudaMemcpy(d_A, A_flat.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B_flat.data(), bytes, cudaMemcpyHostToDevice);

        // Тестирование разных конфигураций блоков
        for (int cfg = 0; cfg < numConfigs; cfg++) {
            int blockX = blockConfigs[cfg].x;
            int blockY = blockConfigs[cfg].y;
            int threads = blockX * blockY;

            dim3 blockDim(blockX, blockY);
            dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                (n + blockDim.y - 1) / blockDim.y);

            // Прогрев
            matrixMulKernel << <gridDim, blockDim >> > (d_A, d_B, d_C, n);
            cudaDeviceSynchronize();

            // Проверка на ошибки
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                cout << "  " << blockConfigs[cfg].name << ": ERROR - " << cudaGetErrorString(err) << endl;
                continue;
            }

            // Измерение времени
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            matrixMulKernel << <gridDim, blockDim >> > (d_A, d_B, d_C, n);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            double speedup = seq_time.count() / elapsed_ms;
            long long ops = 2LL * n * n * n;

            cout << "  " << blockConfigs[cfg].name << ": " << elapsed_ms << " ms (x" << speedup << ")" << endl;

            ofstream results("cuda_all_results.txt", ios::app);
            results << n << "," << blockConfigs[cfg].name << "," << threads << ","
                << elapsed_ms << "," << speedup << "," << ops << endl;
            results.close();
        }

        // Копируем результат для проверки (используем последнюю конфигурацию)
        vector<double> C_flat(size);
        cudaMemcpy(C_flat.data(), d_C, bytes, cudaMemcpyDeviceToHost);

        vector<vector<double>> C(n, vector<double>(n));
        for (int row = 0; row < n; row++)
            for (int col = 0; col < n; col++)
                C[row][col] = C_flat[row * n + col];

        writeResult("cuda_result_matrix.txt", C);

        // Проверка корректности
        bool correct = true;
        for (int row = 0; row < n && correct; row++) {
            for (int col = 0; col < n; col++) {
                if (abs(C[row][col] - C_seq[row][col]) > 1e-5) {
                    correct = false;
                    break;
                }
            }
        }

        if (correct) {
            cout << "  Result: OK" << endl;
        }
        else {
            cout << "  Result: FAIL" << endl;
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    cout << "\n========================================" << endl;
    cout << "All tests completed!" << endl;
    cout << "Results saved to cuda_all_results.txt" << endl;
    cout << "========================================" << endl;

    return 0;
}