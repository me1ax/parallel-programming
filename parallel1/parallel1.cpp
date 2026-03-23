#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>

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

vector<vector<double>> multiply(vector<vector<double>> A, vector<vector<double>> B) {
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

void writeReport(string filename, int n, double time_sec) {
    ofstream file(filename);
    file << "Matrix Size (N): " << n << endl;
    file << "Task Volume (Elements): " << n * n << endl;
    file << "Execution Time (ms): " << time_sec * 1000 << endl;
}

int main() {
    int sizes[] = { 100, 200, 300, 400, 500 };
    int num_sizes = 5;

    ofstream results("all_results.txt");
    results << "N,Time(s),Operations" << endl;

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        cout << "Running N=" << n << "... ";

        generateMatrix("matrix_a.txt", n);
        generateMatrix("matrix_b.txt", n);

        int n1, n2;
        auto A = readMatrix("matrix_a.txt", n1);
        auto B = readMatrix("matrix_b.txt", n2);

        auto start = chrono::high_resolution_clock::now();
        auto C = multiply(A, B);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double> time = end - start;
        long long ops = 2LL * n * n * n;

        writeResult("result_matrix.txt", C);
        writeReport("report.txt", n, time.count());

        results << n << "," << time.count() << "," << ops << endl;
        cout << time.count() << " s" << endl;
    }

    results.close();
    cout << "\nAll results saved to all_results.txt" << endl;
    return 0;
}