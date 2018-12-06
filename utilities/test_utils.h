//
// Created by Martin Galajda on 06/12/2018.
//

#ifndef NEURAL_NETWORKS_TEST_UTILS_H
#define NEURAL_NETWORKS_TEST_UTILS_H

#include "../matrix_impl/Matrix.hpp"
#include "gtest/gtest.h"

void assertSameMatrices(MatrixDoubleSharedPtr expectedMatrix, MatrixDoubleSharedPtr gotMatrix);

void assertSameMatrices(MatrixDoubleSharedPtr expectedMatrix, Matrix<double>  &gotMatrixRef);

#endif //NEURAL_NETWORKS_TEST_UTILS_H
