#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <string>
#include <omp.h>

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

vector<double> readMatrixFlat(string filename, int& n) {
    ifstream file(filename);
    file >> n;
    vector<double> matrix(n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> matrix[i * n + j];
    return matrix;
}

vector<double> multiplyFlat(vector<double>& A, vector<double>& B, int n, int num_threads) {
    vector<double> C(n * n, 0.0);
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
    
    return C;
}

void writeResultFlat(string filename, vector<double>& C, int n) {
    ofstream file(filename);
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            file << C[i * n + j] << " ";
        file << endl;
    }
}

int main() {
    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int threads[] = { 1, 2, 4 };
    
    int max_threads = omp_get_max_threads();
    cout << "Max threads: " << max_threads << endl;
    
    ofstream report("experiment_results.txt");
    
    for (int n : sizes) {
        cout << "Size: " << n << endl;
        
        generateMatrix("a.txt", n);
        generateMatrix("b.txt", n);
        
        int n1, n2;
        auto A = readMatrixFlat("a.txt", n1);
        auto B = readMatrixFlat("b.txt", n2);
        
        for (int t : threads) {
            if (t > max_threads) continue;
            
            auto start = chrono::high_resolution_clock::now();
            auto C = multiplyFlat(A, B, n, t);
            auto end = chrono::high_resolution_clock::now();
            
            chrono::duration<double> time = end - start;
            
            cout << "  Threads " << t << ": " << time.count() << " s" << endl;
            report << n << " " << t << " " << time.count() << endl;
            
            if (n == 2000) {
                writeResultFlat("result_" + to_string(t) + ".txt", C, n);
            }
        }
    }
    
    report.close();
    return 0;
}