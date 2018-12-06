//
// Created by Martin Galajda on 29/10/2018.
//

#include <vector>
#include "../matrix_impl/Matrix.hpp"
#include <memory>

#ifndef UTILITIES_POPULATE_PLACEHOLDERS
#define UTILITIES_POPULATE_PLACEHOLDERS

void populatePlaceholders(
        std::shared_ptr<Matrix<double>> &placeholders,
        std::vector<std::shared_ptr<Matrix<double>>> &pool,
        std::vector<int> &poolIndices
);

void populatePlaceholders2D(
        std::shared_ptr<Matrix<double>> &placeholders,
        std::vector<std::shared_ptr<Matrix<double>>> &pool,
        std::vector<int> &poolIndices
);

#endif //UTILITIES_POPULATE_PLACEHOLDERS
