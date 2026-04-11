#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <mpi.h>
#include <string>

using namespace std;

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

// Функция для запуска теста с определенным количеством процессов
void runTest(int n, int num_procs, int rank, MPI_Comm comm) {
    if (rank == 0) {
        cout << "  Running N=" << n << "... ";
        generateMatrix("matrix_a.txt", n);
        generateMatrix("matrix_b.txt", n);
    }

    MPI_Barrier(comm);

    int n1, n2;
    vector<vector<double>> A, B;

    if (rank == 0) {
        A = readMatrix("matrix_a.txt", n1);
        B = readMatrix("matrix_b.txt", n2);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, comm);

    if (rank != 0) {
        B.resize(n, vector<double>(n));
    }

    for (int row = 0; row < n; row++) {
        MPI_Bcast(B[row].data(), n, MPI_DOUBLE, 0, comm);
    }

    int rows_per_proc = n / num_procs;
    int remainder = n % num_procs;
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    vector<vector<double>> local_A(local_rows, vector<double>(n));

    if (rank == 0) {
        int offset = 0;
        for (int p = 0; p < num_procs; p++) {
            int p_rows = rows_per_proc + (p < remainder ? 1 : 0);
            if (p == 0) {
                for (int i = 0; i < p_rows; i++)
                    for (int j = 0; j < n; j++)
                        local_A[i][j] = A[offset + i][j];
            }
            else {
                for (int i = 0; i < p_rows; i++)
                    MPI_Send(A[offset + i].data(), n, MPI_DOUBLE, p, 0, comm);
            }
            offset += p_rows;
        }
    }
    else {
        for (int i = 0; i < local_rows; i++)
            MPI_Recv(local_A[i].data(), n, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(comm);

    double start_time = MPI_Wtime();

    vector<vector<double>> local_C(local_rows, vector<double>(n, 0.0));
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }

    double end_time = MPI_Wtime();
    double parallel_time = end_time - start_time;

    if (rank == 0) {
        vector<vector<double>> C(n, vector<double>(n));
        int offset = 0;
        for (int p = 0; p < num_procs; p++) {
            int p_rows = rows_per_proc + (p < remainder ? 1 : 0);
            if (p == 0) {
                for (int i = 0; i < p_rows; i++)
                    for (int j = 0; j < n; j++)
                        C[offset + i][j] = local_C[i][j];
            }
            else {
                vector<vector<double>> temp(p_rows, vector<double>(n));
                for (int i = 0; i < p_rows; i++)
                    MPI_Recv(temp[i].data(), n, MPI_DOUBLE, p, 0, comm, MPI_STATUS_IGNORE);
                for (int i = 0; i < p_rows; i++)
                    for (int j = 0; j < n; j++)
                        C[offset + i][j] = temp[i][j];
            }
            offset += p_rows;
        }

        long long ops = 2LL * n * n * n;
        string filename = "result_matrix_" + to_string(n) + "_" + to_string(num_procs) + "p.txt";
        writeResult(filename, C);

        ofstream results("all_results.txt", ios::app);
        results << n << "," << num_procs << "," << parallel_time << "," << ops << endl;
        results.close();

        cout << parallel_time << " s" << endl;
    }
    else {
        for (int i = 0; i < local_rows; i++)
            MPI_Send(local_C[i].data(), n, MPI_DOUBLE, 0, 0, comm);
    }

    MPI_Barrier(comm);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int num_sizes = 6;
    int procs_list[] = { 1, 2, 4, 8 };
    int num_procs_tests = 4;

    if (rank == 0) {
        cout << "========================================" << endl;
        cout << "MPI Matrix Multiplication Benchmark" << endl;
        cout << "========================================" << endl;

        // Очищаем файл результатов
        ofstream results("all_results.txt");
        results << "N,Processes,Time(s),Operations" << endl;
        results.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Для каждого количества процессов
    for (int p_idx = 0; p_idx < num_procs_tests; p_idx++) {
        int target_procs = procs_list[p_idx];

        // Создаем коммуникатор только для нужного количества процессов
        MPI_Comm test_comm;
        MPI_Comm_split(MPI_COMM_WORLD, rank < target_procs ? 1 : MPI_UNDEFINED, rank, &test_comm);

        if (rank < target_procs) {
            int test_rank, test_size;
            MPI_Comm_rank(test_comm, &test_rank);
            MPI_Comm_size(test_comm, &test_size);

            if (test_rank == 0) {
                cout << "\n=== Testing with " << test_size << " processes ===" << endl;
            }

            // Для каждого размера матрицы
            for (int i = 0; i < num_sizes; i++) {
                runTest(sizes[i], test_size, test_rank, test_comm);
            }

            MPI_Comm_free(&test_comm);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        cout << "\n========================================" << endl;
        cout << "All tests completed!" << endl;
        cout << "Results saved to all_results.txt" << endl;
        cout << "========================================" << endl;
    }

    MPI_Finalize();
    return 0;
}