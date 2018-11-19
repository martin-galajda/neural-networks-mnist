//
// Created by Martin Galajda on 29/10/2018.
//
#define _GLIBCXX_USE_CXX11_ABI 0

#include "ComputationalGraph.h"
#include <random>
#include "./BaseOptimizer.h"

#ifndef MATRIXBENCHMARKS_MINIBATCHOPTIMIZER_H
#define MATRIXBENCHMARKS_MINIBATCHOPTIMIZER_H

class MiniBatchOptimizer: public BaseOptimizer {
public:
    using BaseOptimizer::BaseOptimizer;
    virtual void train();
};


#endif //MATRIXBENCHMARKS_MINIBATCHOPTIMIZER_H
