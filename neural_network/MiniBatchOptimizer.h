//
// Created by Martin Galajda on 29/10/2018.
//

#include "ComputationalGraph.h"
#include <random>
#include "./BaseOptimizer.h"

#ifndef NEURAL_NETWORKS_MINIBATCHOPTIMIZER_H
#define NEURAL_NETWORKS_MINIBATCHOPTIMIZER_H

class MiniBatchOptimizer: public BaseOptimizer {
public:
    using BaseOptimizer::BaseOptimizer;
    virtual void train();
};


#endif //NEURAL_NETWORKS_MINIBATCHOPTIMIZER_H
