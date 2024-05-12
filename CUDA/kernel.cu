#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <__msvc_chrono.hpp>

int* GenerateRandomArray(int n) {
    int* arr = new int[n];
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % 10;
    }
    return arr;
}

double f(double x, double y) {
    return x * x + y * y; // Пример функции x^2 + y^2
}

__global__ void hello_world() {
    printf("Hello World\n");
}

// CUDA kernel для вычисления суммы массива чисел
__global__ void calculateSum(int* array, int size, int* sum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tempSum = 0;

    while (tid < size) {
        tempSum += array[tid];
        tid += blockDim.x * gridDim.x;
    }

    atomicAdd(sum, tempSum);
}

__global__ void calculatePartialDerivative(double* A, double* B, int rows, int cols, double h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;

        // Частная производная df/dx = (f(x + h, y) - f(x, y)) / h
        if (idx > 0 && idx < cols - 1) {
            B[index] = (A[index + 1] - A[index - 1]) / (2 * h);
        }
        else {
            B[index] = 0.0; // Граничные условия
        }
    }
}

__global__ void matrixMultiplication(int* A, int* B, int* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        int sum = 0;
        for (int i = 0; i < colsA; ++i) {
            sum += A[row * colsA + i] * B[i * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}


void Task2(int arraySize) {
    std::cout << "Task2 : " << arraySize << " elements" << std::endl;
    auto start = std::chrono::steady_clock::now();

    const int blockSize = 256; // Размер блока нитей CUDA
    const int numBlocks = (arraySize + blockSize - 1) / blockSize; // Количество блоков CUDA

    int* array = GenerateRandomArray(arraySize); // Создание массива
    int* d_array; // Указатель на массив на устройстве (GPU)
    int* d_sum; // Указатель на сумму на устройстве (GPU)
    int sum = 0; // Сумма производного массива

    // Выделение памяти на устройстве (GPU)
    cudaMalloc(&d_array, arraySize * sizeof(int));
    cudaMalloc(&d_sum, sizeof(int));

    // Копирование данных из хоста (CPU) в устройство (GPU)
    cudaMemcpy(d_array, array, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &sum, sizeof(int), cudaMemcpyHostToDevice);

    // Вызов CUDA kernel для вычисления суммы производного массива чисел
    calculateSum << <numBlocks, blockSize >> > (d_array, arraySize, d_sum);

    // Копирование данных с устройства (GPU) на хост (CPU)
    cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    // Вывод результата
    std::cout << "Sum of array: " << sum << std::endl;

    // Освобождение выделенной памяти на устройстве (GPU)
    cudaFree(d_array);
    cudaFree(d_sum);

    // Освобождение памяти на хосте (CPU)
    delete[] array;
    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

void Task3(int arraySize) {
    std::cout << "Task3 : " << arraySize * arraySize << " elements" << std::endl;
    auto start = std::chrono::steady_clock::now();
    int rows = arraySize;
    int cols = arraySize;
    const double h = 0.01;

    // Выделение памяти на хосте
    double* A = new double[rows * cols];
    double* B = new double[rows * cols];

    // Инициализация массива A
    for (int i = 0; i < rows * cols; ++i) {
        int row = i / cols;
        int col = i % cols;
        A[i] = f(row, col); // Функция f(x, y) = x^2 + y^2
    }

    // Выделение памяти на устройстве
    double* d_A, * d_B;
    cudaMalloc(&d_A, rows * cols * sizeof(double));
    cudaMalloc(&d_B, rows * cols * sizeof(double));

    // Копирование данных из хоста на устройство
    cudaMemcpy(d_A, A, rows * cols * sizeof(double), cudaMemcpyHostToDevice);

    // Запуск CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    calculatePartialDerivative << <numBlocks, threadsPerBlock >> > (d_A, d_B, rows, cols, h);

    // Копирование данных с устройства на хост
    cudaMemcpy(B, d_B, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Освобождение памяти на устройстве и хосте
    cudaFree(d_A);
    cudaFree(d_B);
    delete[] A;
    delete[] B;
    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

void Task4(int rowsA, int colsA, int colsB) {
    std::cout << "Task4 : " << rowsA * colsA << " elements" << std::endl;
    auto start = std::chrono::steady_clock::now();
    const int rowsB = colsA;

    // Выделение памяти на хосте
    int* A = new int[rowsA * colsA];
    int* B = new int[rowsB * colsB];
    int* C = new int[rowsA * colsB];

    // Инициализация матриц A и B
    for (int i = 0; i < rowsA * colsA; ++i) {
        A[i] = i;
    }
    for (int i = 0; i < rowsB * colsB; ++i) {
        B[i] = i;
    }

    // Выделение памяти на устройстве
    int* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, rowsA * colsA * sizeof(int));
    cudaMalloc(&d_B, rowsB * colsB * sizeof(int));
    cudaMalloc(&d_C, rowsA * colsB * sizeof(int));

    // Копирование данных из хоста на устройство
    cudaMemcpy(d_A, A, rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, rowsB * colsB * sizeof(int), cudaMemcpyHostToDevice);

    // Запуск CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((colsB + threadsPerBlock.x - 1) / threadsPerBlock.x, (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMultiplication << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, rowsA, colsA, colsB);

    // Копирование данных с устройства на хост
    cudaMemcpy(C, d_C, rowsA * colsB * sizeof(int), cudaMemcpyDeviceToHost);

    // Освобождение памяти на устройстве и хосте
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C;

    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

int main()
{
    std::cout << "Cuda Implementation" << std::endl;
    hello_world << <1, 1 >> > ();
    
    Task2(10);
    Task2(1000);
    Task2(100000);
    Task2(10000000);
    Task2(100000000);

    Task3(10);
    Task3(100);
    Task3(1000);
    Task3(10000);

    Task4(10, 10, 10);
    Task4(100, 100, 100);
    Task4(1000, 1000, 1000);
    Task4(10000, 10000, 10000);

    return 0;
}
