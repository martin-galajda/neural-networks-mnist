//
// Created by Martin Galajda on 29/10/2018.
//
#define _GLIBCXX_USE_CXX11_ABI 0

#include "./populate_placeholders.h"

void populatePlaceholders(
        std::shared_ptr<Matrix<double>> &placeholders,
        std::vector<std::shared_ptr<Matrix<double>>> &pool,
        std::vector<int> &poolIndices
) {
    for (auto iteration = 0; iteration < poolIndices.size(); iteration++) {
        auto pickedInstanceIndex = poolIndices[iteration];
        auto instance = pool[pickedInstanceIndex];

        for (auto row = 0; row < instance->getNumOfRows(); row++) {
            (*placeholders)[row][iteration] = (*instance)[row][0];
        }
    }
}
