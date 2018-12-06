//
// Created by Martin Galajda on 28/11/2018.
//

#ifndef NEURAL_NETWORKS_CONVOLUTION_H
#define NEURAL_NETWORKS_CONVOLUTION_H

#include "../matrix_impl/Matrix.hpp"

bool is_integer(float k);
//MatrixDoubleSharedPtr convolution(MatrixDoubleSharedPtr &input, MatrixDoubleSharedPtr &kernel, int stride);
MatrixDouble *convolution(MatrixDoubleSharedPtr &input, MatrixDouble *output, MatrixDouble *kernel, int stride);


#endif //NEURAL_NETWORKS_CONVOLUTION_H
