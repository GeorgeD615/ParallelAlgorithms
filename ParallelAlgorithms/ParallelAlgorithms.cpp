#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <ctime>
#include <thread>

namespace SimpleImplementation {
    void Task1() {
        std::cout << "Task1" << std::endl;
        auto start = std::chrono::steady_clock::now();

        std::cout << "Hello world" << std::endl;

        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
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

}

namespace OpenMP {
    void Task1(){
        std::cout << "Task1_OMP" << std::endl;
        auto start = std::chrono::steady_clock::now();

        #pragma omp parallel
        {
            std::cout << "Hello world from thread " << omp_get_thread_num() << std::endl;
        }

        auto end = std::chrono::steady_clock::now();
        std::cout <<"Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
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
}

//Command to run an application on 4 mpi processors:
//mpiexec -n 4 ParallelAlgorithms 
namespace MPI {

    void Task1(int size, int rank) {
        std::cout << "Task1_MPI" << std::endl;
        auto start = std::chrono::steady_clock::now();
        std::cout << "Hello world from processor " << rank <<  " out of " << size << " processors." << std::endl;
        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    void Task2(int* arr, int Length, int rank, int size) {
        std::cout << "Task2_MPI : " << Length << " elements" << std::endl;
        auto start = std::chrono::steady_clock::now();

        int n = Length;

        // Temporary array for slave process
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

                // последний процесс подсчитывает оставшиеся элементы
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

        // cleans up all MPI state before exit of process


        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
}

int* GenerateRandomArray(int n) {
    int* arr = new int[n];
    #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            arr[i] = rand() % 10;
        }
    return arr;
}

int main()
{
    std::cout << "------Simple implementation------" << std::endl;

    SimpleImplementation::Task1();

    SimpleImplementation::Task2(GenerateRandomArray(10), 10);
    SimpleImplementation::Task2(GenerateRandomArray(1000), 1000);
    SimpleImplementation::Task2(GenerateRandomArray(10000000), 10000000);

    std::cout << std::endl;

    std::cout << "------OpenMP implementation------" << std::endl;

    OpenMP::Task1();

    OpenMP::Task2(GenerateRandomArray(10), 10);
    OpenMP::Task2(GenerateRandomArray(1000), 1000);
    OpenMP::Task2(GenerateRandomArray(10000000), 10000000);

    std::cout << std::endl;

    std::cout << "------MPI implementation------" << std::endl;

    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "Size : " << size << std::endl;

    MPI::Task1(rank, size);
     
    MPI::Task2(GenerateRandomArray(10), 10, rank, size);
    MPI::Task2(GenerateRandomArray(1000), 1000, rank, size);
    MPI::Task2(GenerateRandomArray(10000000), 10000000, rank, size);

    MPI_Finalize();

    std::cout << std::endl;

    
}
