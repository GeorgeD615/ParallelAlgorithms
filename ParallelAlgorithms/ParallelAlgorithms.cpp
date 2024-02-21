#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <ctime>
#include <thread>


namespace OpenMP {
    void Task1_OMP(){
        std::cout << "Task1_OMP" << std::endl;
        auto start = std::chrono::steady_clock::now();

        #pragma omp parallel
        {
            std::cout << "Hello world" << std::endl;
        }

        auto end = std::chrono::steady_clock::now();
        std::cout <<"Passed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    void Task2_OMP(int* arr, int Length) {
        std::cout << "Task2_OMP : " << Length << " elements" << std::endl;
        auto start = std::chrono::steady_clock::now();

        int result = 0;
        #pragma omp parallel for
            for (int i = 0; i < Length; ++i) {
                result += arr[i];
            }

        auto end = std::chrono::steady_clock::now();
        std::cout << "Passed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
}

namespace MPI {
    void Task1_MPI() {
        MPI_Init(NULL, NULL);
        int rank, size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cout << "Hello world size : " << size << " rank : " << rank << std::endl;
        MPI_Finalize();
    }
}

int* GenerateRandomArray(int n) {
    int* arr = new int[n];
    #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            arr[i] = rand();
        }
    return arr;
}

int main()
{
    OpenMP::Task1_OMP();
    OpenMP::Task2_OMP(GenerateRandomArray(10), 10);
    OpenMP::Task2_OMP(GenerateRandomArray(1000), 1000);
    OpenMP::Task2_OMP(GenerateRandomArray(10000000), 10000000);
    MPI::Task1_MPI();
}
