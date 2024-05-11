#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <ctime>
#include <thread>

double f(double x, double y) {
    return x * x + y * y; // Пример функции x^2 + y^2
}
// Функция для вычисления частной производной df/dx
double df_dx(double x, double y, double h) {
    return (f(x + h, y) - f(x, y)) / h;
}

int* GenerateRandomArray(int n) {
    int* arr = new int[n];
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % 10;
    }
    return arr;
}

// Метод для генерации матрицы случайных чисел заданного размера
int** GenerateRandomMatrix(int rows, int cols) {
    int** matrix = new int* [rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new int[cols];
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = rand() % 10;
        }
    }
    return matrix;
}

void DeleteMatrix(int** matrix, int rows) {
    for (int i = 0; i < rows; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

//Implementation without frameworks
namespace SimpleImplementation {
    void Task1() {
        std::cout << "Hello world" << std::endl;
    }

    void Task2(int* arr, int Length) {
        std::cout << "Task2 : " << Length << " elements" << std::endl;
        auto start = std::chrono::steady_clock::now();

        int result = 0;
        for (int i = 0; i < Length; ++i) {
            result += arr[i];
        }

        std::cout << "Sum of array is : " << result << std::endl;
        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    void Task3(int rows, int cols) {
        std::cout << "Task3 : " << rows*cols << " elements" << std::endl;
        auto start = std::chrono::steady_clock::now();

        const double h = 0.01; // Шаг сетки
        double** B = new double* [rows]; // Массив для хранения значений производной
        for (int i = 0; i < rows; ++i) {
            B[i] = new double[cols];
        }
        // Вычисление производной и заполнение массива B
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                B[i][j] = df_dx(i, j, h); // df/dx
            }
        }

        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    void Task4(int** A, int** B, int rowsA, int colsA, int colsB) {
        std::cout << "Task4 : " << rowsA * colsA << " elements" << std::endl;
        auto start = std::chrono::steady_clock::now();

        int** result = new int* [rowsA];
        for (int i = 0; i < rowsA; ++i) {
            result[i] = new int[colsB];
            for (int j = 0; j < colsB; ++j) {
                result[i][j] = 0;
                for (int k = 0; k < colsA; ++k) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        DeleteMatrix(result, rowsA);

        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
}

//Implementation with OpenMP framework
namespace OpenMP {
    void Task1(){
        #pragma omp parallel
        {
            std::cout << "Hello world from thread " << omp_get_thread_num() << std::endl;
        }
    }

    void Task2(int* arr, int Length) {
        std::cout << "Task2_OMP : " << Length << " elements" << std::endl;
        auto start = std::chrono::steady_clock::now();

        int result = 0;
        #pragma omp parallel for
            for (int i = 0; i < Length; ++i) {
                result += arr[i];
            }
        std::cout << "Sum of array is : " << result << std::endl;
        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    void Task3(int rows, int cols) {
        std::cout << "Task3_OMP : " << rows * cols << " elements" << std::endl;
        auto start = std::chrono::steady_clock::now();

        const double h = 0.01; // Шаг сетки
        double** B = new double* [rows]; // Массив для хранения значений производной
        for (int i = 0; i < rows; ++i) {
            B[i] = new double[cols];
        }

        // Вычисление производной и заполнение массива B
        #pragma omp parallel for
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                B[i][j] = df_dx(i, j, h); // df/dx
            }
        }

        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    void Task4(int** A, int** B, int rowsA, int colsA, int colsB) {
        std::cout << "Task4 : " << rowsA * colsA << " elements" << std::endl;
        auto start = std::chrono::steady_clock::now();

        int** result = new int* [rowsA];
        for (int i = 0; i < rowsA; ++i) {
            result[i] = new int[colsB];
            for (int j = 0; j < colsB; ++j) {
                result[i][j] = 0;
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsB; ++j) {
                for (int k = 0; k < colsA; ++k) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        DeleteMatrix(result, rowsA);

        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    }
}

//Implementation with MPI framework
//Command to run an application on 4 mpi processors:
//mpiexec -n 4 ParallelAlgorithms 
namespace MPI {

    void Task1(int rank, int size) {
        std::cout << "Hello world from processor " << rank <<  " out of " << size << " processors." << std::endl;
    }

    void Task2(int* arr, int Length, int rank, int size) {

        std::cout << "Task2_MPI : " << Length << " elements" << std::endl;
        auto start = std::chrono::steady_clock::now();

        int n = Length;

        // temp array for secondary process
        int* arr2 = new int[Length / size + 1];

        int elements_per_process, n_elements_recieved;

        MPI_Status status;

        // master process
        if (rank == 0) {
            int index, i;
            elements_per_process = n / size;

            // check if more than 1 processes are run
            if (size > 1) {
                // distributes the portion of array
                // to child processes to calculate
                // their partial sums
                for (i = 1; i < size - 1; i++) {
                    index = i * elements_per_process;
                    MPI_Send(&elements_per_process, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                    MPI_Send(&arr[index], elements_per_process, MPI_INT, i, 0, MPI_COMM_WORLD);
                }

                // last process count left elements
                index = i * elements_per_process;
                int elements_left = n - index;

                MPI_Send(&elements_left, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&arr[index], elements_left, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

            // master process add its own sub array
            int sum = 0;
            for (i = 0; i < elements_per_process; i++)
                sum += arr[i];

            // collects partial sums from other processes
            int tmp;
            for (i = 1; i < size; i++) {
                MPI_Recv(&tmp, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                int sender = status.MPI_SOURCE;
                sum += tmp;
            }

            // prints the final sum of array
            std::cout << "Sum of array is : " << sum << std::endl;
        }
        // slave processes
        else {
            MPI_Recv(&n_elements_recieved, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            // stores the received array segment
            // in local array a2
            MPI_Recv(arr2, n_elements_recieved, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            // calculates its partial sum
            int partial_sum = 0;
            for (int i = 0; i < n_elements_recieved; i++) partial_sum += arr2[i];

            // sends the partial sum to the root process
            MPI_Send(&partial_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    void Task3(int rows, int cols, int rank, int size) {
        std::cout << "Task3_MPI : " << rows*cols << " elements" << std::endl;
        auto start = std::chrono::steady_clock::now();

        int n = rows;

        // temp array for secondary process
        double* arr2 = new double[rows / (size - 1) * cols];
        double h = 0.01;

        int elements_per_process, n_elements_recieved;

        MPI_Status status;

        // master process
        if (rank == 0) {
            double** B = new double* [rows]; // Массив для хранения значений производной
            for (int i = 0; i < rows; ++i) {
                B[i] = new double[cols];
            }

            // collects subarrays from other processes
            double* tmp = new double[rows/(size-1) * cols];
            int j;
            int l;
            for (int i = 1; i < size - 1; i++) {
                MPI_Recv(tmp, rows / (size - 1) * cols, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, & status);
                
                l = 0;
                for (j = (i - 1) * (rows / (size - 1)); j < i * rows / (size - 1); ++j) {
                    for (int k = 0; k < cols; ++k) {
                        B[j][k] = tmp[l * cols + k];
                    }
                    ++l;
                }
            }

            MPI_Recv(tmp, rows / (size - 1) * cols, MPI_DOUBLE, size - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            l = 0;
            for (int i = j; i < rows; ++i) {
                for (int k = 0; k < cols; ++k) {
                    B[i][k] = tmp[l * cols + k];
                }
                ++l;
            }
        }
        // slave processes
        else {
            for (int i = 0; i < rows / (size - 1); ++i) {
                for (int j = 0; j < cols; ++j) {
                    arr2[i * cols + j] = df_dx(i + (rank-1) * rows / (size - 1), j, h); // df/dx
                }
            }

            // sends the partial sum to the root process
            MPI_Send(arr2, rows / (size - 1) * cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }

        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    void Task4(int** A, int** B, int rowsA, int colsA, int colsB, int rank, int size) {
        std::cout << "Task4_MPI : " << rowsA * colsA << " elements" << std::endl;
        auto start = std::chrono::steady_clock::now();

        int n = rowsA;

        // temp array for secondary process
        int* arr2 = new int[n / (size - 1) * n];
        for (int i = 0; i < n / (size - 1) * n; ++i) {
            arr2[i] = 0;
        }

        MPI_Status status;

        // master process
        if (rank == 0) {
            int** C = new int* [n];
            for (int i = 0; i < n; ++i) {
                C[i] = new int[n];
                for (int j = 0; j < n; ++j) {
                    C[i][j] = 0;
                }
            }

            // collects subarrays from other processes
            int* tmp = new int[n / (size - 1) * n];
            int j;
            int l;
            for (int i = 1; i < size - 1; i++) {
                MPI_Recv(tmp, n / (size - 1) * n, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                l = 0;
                for (j = (i - 1) * (n / (size - 1)); j < i * n / (size - 1); ++j) {
                    for (int k = 0; k < n; ++k) {
                        C[j][k] = tmp[l * n + k];
                    }
                    ++l;
                }
            }

            MPI_Recv(tmp, n / (size - 1) * n, MPI_DOUBLE, size - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            l = 0;
            for (int i = j; i < n; ++i) {
                for (int k = 0; k < n; ++k) {
                    C[i][k] = tmp[l * n + k];
                }
                ++l;
            }
        }
        // slave processes
        else {
            if (rank != size - 1) {
                for (int k = 0; k < n; k++)
                    for (int i = (rank-1) * n / (size - 1); i < rank * n / (size - 1); i++) {
                        for (int j = 0; j < n; j++)
                            arr2[i * n + j] = arr2[i * n + j] + A[i][j] * B[j][k];
                    }
            }
            else {
                for (int k = 0; k < n; k++)
                    for (int i = (rank - 1) * n; i < n; i++) {
                        for (int j = 0; j < n; j++)
                            arr2[i * n + j] = arr2[i * n + j] + A[i][j] * B[j][k];
                    }
            }
            // sends the partial sum to the root process
            MPI_Send(arr2, n / (size - 1) * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }

        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
}


int main()
{
    /*std::cout << "------Simple implementation------" << std::endl;

    SimpleImplementation::Task1();

    SimpleImplementation::Task2(GenerateRandomArray(10), 10);
    SimpleImplementation::Task2(GenerateRandomArray(1000), 1000);
    SimpleImplementation::Task2(GenerateRandomArray(10000000), 10000000);

    SimpleImplementation::Task3(10, 10);
    SimpleImplementation::Task3(100, 100);
    SimpleImplementation::Task3(1000, 1000);

    SimpleImplementation::Task4(GenerateRandomMatrix(10, 10), GenerateRandomMatrix(10, 10), 10, 10, 10);
    SimpleImplementation::Task4(GenerateRandomMatrix(100, 100), GenerateRandomMatrix(100, 100), 100, 100, 100);
    SimpleImplementation::Task4(GenerateRandomMatrix(1000, 1000), GenerateRandomMatrix(1000, 1000), 1000, 1000, 1000);


    std::cout << std::endl;

    std::cout << "------OpenMP implementation------" << std::endl;

    OpenMP::Task1();

    OpenMP::Task2(GenerateRandomArray(10), 10);
    OpenMP::Task2(GenerateRandomArray(1000), 1000);
    OpenMP::Task2(GenerateRandomArray(10000000), 10000000);

    OpenMP::Task3(10, 10);
    OpenMP::Task3(100, 100);
    OpenMP::Task3(1000, 1000);

    OpenMP::Task4(GenerateRandomMatrix(10, 10), GenerateRandomMatrix(10, 10), 10, 10, 10);
    OpenMP::Task4(GenerateRandomMatrix(100, 100), GenerateRandomMatrix(100, 100), 100, 100, 100);
    OpenMP::Task4(GenerateRandomMatrix(1000, 1000), GenerateRandomMatrix(1000, 1000), 1000, 1000, 1000);*/

    //std::cout << std::endl;

    //std::cout << "------MPI implementation------" << std::endl;
    //MPI_Init(NULL, NULL);
    //int rank, size;
    //MPI_Comm_size(MPI_COMM_WORLD, &size);
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //MPI::Task1(rank, size);
    //
    // на 4х потоках 
    //MPI::Task2(GenerateRandomArray(10), 10, rank, size);
    //MPI::Task2(GenerateRandomArray(1000), 1000, rank, size);
    //MPI::Task2(GenerateRandomArray(10000000), 10000000, rank, size);

    //на 5и потоках
    //MPI::Task3(10, 10, rank, size);
    //MPI::Task3(100, 100, rank, size);
    //MPI::Task3(1000, 1000, rank, size);

    //на 3х потоках
    //MPI::Task4(GenerateRandomMatrix(16, 16), GenerateRandomMatrix(16, 16), 16, 16, 16, rank, size);
    //MPI::Task4(GenerateRandomMatrix(128, 128), GenerateRandomMatrix(128, 128), 128, 128, 128, rank, size);
    //MPI::Task4(GenerateRandomMatrix(1024, 1024), GenerateRandomMatrix(1024, 1024), 1024, 1024, 1024, rank, size);
    //MPI::Task4(GenerateRandomMatrix(2048, 2048), GenerateRandomMatrix(2048, 2048), 2048, 2048, 2048, rank, size);

    //MPI_Finalize();

    //std::cout << std::endl;
}
