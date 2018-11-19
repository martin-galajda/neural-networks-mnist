//
// Created by Martin Galajda on 17/10/2018.
//

#include "gtest/gtest.h"
#include "../matrix_impl/Matrix.hpp"
#include "Matrix.cpp"
#include <cmath>

bool double_equals(double a, double b, double epsilon = 0.01)
{
    return std::abs(a - b) < epsilon;
}

TEST(matrixTests, dummyTest)
{
    EXPECT_EQ(1000, 1000);
}

TEST(matrixTests, multiplication)
{
    // A = | 1 2 3 |
    //     | 1 2 3 |
    double dataForA[] = { 1, 2, 3,
                          1, 2, 3 };
    Matrix<double> *A = new Matrix<double>(dataForA, 2, 3);

    // B = | 1 2 1 2 |
    //     | 3 4 3 4 |
    //     | 5 6 5 6 |
    double dataForB[] = { 1, 2, 1, 2,
                          3, 4, 3, 4,
                          5, 6, 5, 6 };
    Matrix<double> *B = new Matrix<double>(dataForB, 3, 4);

    // C = A * B = | 22 | 28 | 22 | 28 |
    //             | 22 | 28 | 22 | 28 |
    // C = (2,4)
    Matrix<double> *C = *A * *B;

    EXPECT_EQ(C->getNumOfRows(), A->getNumOfRows());
    EXPECT_EQ(C->getNumOfCols(), B->getNumOfCols());

    double expectedValues[] = { 22, 28, 22, 28 };

    for (auto i = 0; i < C->getNumOfRows(); i++) {
        for (auto j = 0; j < C->getNumOfCols(); j++) {
            EXPECT_DOUBLE_EQ((*C)[i][j], expectedValues[j]);
        }
    }

    free(A);
    free(B);
    free(C);
}


TEST(matrixTests, transpose)
{
    // A = | 1 2 3 |
    //     | 4 5 6 |
    double dataForA[] = { 1, 2, 3,
                          4, 5, 6 };
    Matrix<double> *A = new Matrix<double>(dataForA, 2, 3);

    auto *B= A->transposeToNew();
    double expectedValues[] = { 1, 4, 2, 5, 3, 6};

    for (auto i = 0; i < B->getNumOfRows(); i++) {
        for (auto j = 0; j < B->getNumOfCols(); j++) {

            auto expectedValue = expectedValues[(i * B->getNumOfCols()) + j];
            auto gotValue = (*B)[i][j];

            EXPECT_DOUBLE_EQ(gotValue, expectedValue);
        }
    }

    free(A);
    free(B);
}

TEST(matrixTests, softmax)
{
    // A = | 1 2 3 |
    //     | 1 2 3 |
    double dataForA[] = { 1, 1, 1,
                          1, 1, 1 };
    Matrix<double> *A = new Matrix<double>(dataForA, 3, 2);

    auto *softmaxResult = A->softmax();

    double expectedValues[] = { 1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3 };

    for (auto i = 0; i < softmaxResult->getNumOfRows(); i++) {
        for (auto j = 0; j < softmaxResult->getNumOfCols(); j++) {
            auto result = (*softmaxResult)[i][j];
            auto expectedValue = expectedValues[(i * softmaxResult->getNumOfCols()) + j];
            EXPECT_NEAR(result, expectedValue, 0.01);
        }
    }

    free(A);
    free(softmaxResult);
}

TEST(matrixTests, softmax2)
{
    // A = | 1 2 3 |
    //     | 1 2 3 |
    double dataForA[] = {
            3, 3,
            4, 4,
            1, 1
    };
    Matrix<double> *A = new Matrix<double>(dataForA, 3, 2);

    auto *softmaxResult = A->softmax();

    double expectedValues[] = {
            0.25949646034242, 0.25949646034242,
            0.70538451269824, 0.70538451269824,
            0.03511902695934, 0.03511902695934
    };

    for (auto i = 0; i < softmaxResult->getNumOfRows(); i++) {
        for (auto j = 0; j < softmaxResult->getNumOfCols(); j++) {
            auto result = (*softmaxResult)[i][j];
            EXPECT_NEAR(result, expectedValues[(i * softmaxResult->getNumOfCols()) + j], 0.01);
        }
    }

    free(A);
    free(softmaxResult);
}


TEST(matrixTests, relu)
{
    // A = | 1 2 3 |
    //     | 1 2 3 |
    double dataForA[] = {
            -3, 3,
            1, -4,
            0, 1
    };
    Matrix<double> *A = new Matrix<double>(dataForA, 3, 2);

    auto *reluResult = A->relu();

    double expectedValues[] = {
            0, 3,
            1, 0,
            0, 1
    };

    for (auto i = 0; i < reluResult->getNumOfRows(); i++) {
        for (auto j = 0; j < reluResult->getNumOfCols(); j++) {
            auto result = (*reluResult)[i][j];
            EXPECT_NEAR(result, expectedValues[(i * reluResult->getNumOfCols()) + j], 0.01);
        }
    }

    free(A);
    free(reluResult);
}
