//
// Created by Martin Galajda on 21/10/2018.
//

#include <iostream>
#include <vector>

#include "benchmark_impl/MatrixV1.hpp"
#include "benchmark_impl/MatrixV3.hpp"
#include "benchmark_impl/MatrixV4.hpp"
#include "benchmark_impl/MatrixV5.hpp"
#include "benchmark_impl/MatrixV6.hpp"
#include "benchmark_impl/MatrixV7.hpp"

double * generateDataForMatrix(int numOfRows, int numOfCols) {
    auto i = 0;
    double *data = (double*) malloc(sizeof(double) * numOfRows * numOfCols);
    for (auto row = 0; row < numOfRows; row++) {
        for (auto col = 0; col < numOfCols; col++) {
            *(data + i) = i;
            i += 1;
        }
    }

    return data;
}


template <class MatrixImpl>
double* benchmarkMatrixInitAndMult(double *dummyData) {
    auto t_start = std::chrono::high_resolution_clock::now();

    auto &matrix1 = *(new MatrixImpl(dummyData, 1000, 1000));
    auto &matrix2 = *(new MatrixImpl(dummyData, 1000, 1000));
    MatrixImpl *matrix3 = matrix1 * matrix2;

    auto t_end = std::chrono::high_resolution_clock::now();


    // Get the dimensions from resulting matrix so the code optimizer doesn't cut off the dead code
    double v1 = matrix3->getNumOfRows();
    double v2 = matrix3->getNumOfCols();
    double ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    free(matrix3);

    double *results = new double[3];

    results[0] = ms;
    results[1] = v1;
    results[2] = v2;

    return results;
}

template <class MatrixImpl>
void runBenchmarksForMatrixInitAndMult(double *dummyData, std::string version, int times = 1) {
    auto results = 0;
    int dimensionFirst = 0;
    int dimensionSecond = 0;

    double* benchmarkResult = 0;
    for (auto i = 0; i < times; i++) {
        benchmarkResult = benchmarkMatrixInitAndMult<MatrixImpl>(dummyData);
        results += benchmarkResult[0];
        dimensionFirst = benchmarkResult[1];
        dimensionSecond = benchmarkResult[2];

        free(benchmarkResult);
    }

    auto avgTime = results / times;
    std::cout << "Took on average " << avgTime << " ms to perform benchmark for init and multiplication for matrix impl: " << version << std::endl;
    std::cout << "Result dimensions: " <<  dimensionFirst << ", " << dimensionSecond << std::endl;
}


template <class MatrixImpl>
double* benchmarkMatrixMult(MatrixImpl &A, MatrixImpl &B) {
    auto t_start = std::chrono::high_resolution_clock::now();

    auto C = A * B;

    auto t_end = std::chrono::high_resolution_clock::now();

    // Get the dimensions from resulting matrix so the code optimizer doesn't cut off the dead code
    double v1 = C->getNumOfRows();
    double v2 = C->getNumOfCols();
    double ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();


    double *results = new double[3];
    results[0] = ms;
    results[1] = v1;
    results[2] = v2;

    free(C);

    return results;
}


template <class MatrixImpl>
void runBenchmarksForMatrixMult(double *dummyData, std::string version, int times = 1) {
    auto results = 0.0;
    int firstDimension = 0;
    int secondDimension = 0;

    auto *A = new MatrixImpl(dummyData, 1000, 1000);
    auto *B = new MatrixImpl(dummyData, 1000, 1000);

    double* benchmarkResult = 0;
    for (auto i = 0; i < times; i++) {
        benchmarkResult = benchmarkMatrixMult<MatrixImpl>(*A,*B);
        results += benchmarkResult[0];
        firstDimension = benchmarkResult[1];
        secondDimension = benchmarkResult[2];
        free(benchmarkResult);
    }

    auto avgTime = results / times;
    std::cout << "Took on average " << avgTime << " ms to perform benchmark for multiplication for matrix impl: " << version << std::endl;
    std::cout << "Result dimensions: " << firstDimension << ", " <<  secondDimension << std::endl;

    free(A);
    free(B);
}

int main() {
    double *dummyData = generateDataForMatrix(10000, 10000);

//    runBenchmarksForMatrixInitAndMult<MatrixV4<double>>(dummyData, "MatrixV4 - allocating one chunk of memory", 5);
//    runBenchmarksForMatrixInitAndMult<Matrix<double>>(dummyData, "Matrix - allocating memory for each row", 5);
//    runBenchmarksForMatrixInitAndMult<MatrixV3<double>>(dummyData, "MatrixV3 - using std::vectors", 5);

    runBenchmarksForMatrixMult<MatrixV6<double>>(dummyData, "MatrixV7 - allocating one chunk of memory + transpose + only raw memory", 5);
    runBenchmarksForMatrixMult<MatrixV6<double>>(dummyData, "MatrixV6 - allocating one chunk of memory + transpose", 5);
    runBenchmarksForMatrixMult<MatrixV5<double>>(dummyData, "MatrixV5 - allocating one chunk of memory + cache efficiency", 5);
    runBenchmarksForMatrixMult<MatrixV4<double>>(dummyData, "MatrixV4 - allocating one chunk of memory", 5);
    runBenchmarksForMatrixMult<MatrixV1<double>>(dummyData, "MatrixV1 - allocating memory for each row", 5);
    runBenchmarksForMatrixMult<MatrixV3<double>>(dummyData, "MatrixV3 - using std::vectors", 5);

    free(dummyData);

    return 0;
}