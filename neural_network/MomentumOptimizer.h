//
// Created by Martin Galajda on 29/10/2018.
//
#define _GLIBCXX_USE_CXX11_ABI 0

#ifndef MATRIXBENCHMARKS_MOMENTUMOPTIMIZER_H
#define MATRIXBENCHMARKS_MOMENTUMOPTIMIZER_H


#include "BaseOptimizer.h"

class MomentumOptimizer: public BaseOptimizer {
public:
    using BaseOptimizer::BaseOptimizer;
    MomentumOptimizer(
            ComputationalGraph &computationalGraph,
            std::vector<std::shared_ptr<Matrix<double>>> &instances,
            std::vector<std::shared_ptr<Matrix<double>>> &labels,
            std::vector<int> &trainIndices,
            int minibatchSize,
            double learningRate
    );

    virtual void train();
protected:

    std::vector<std::shared_ptr<Matrix<double>>> velocities;
    std::vector<std::shared_ptr<Matrix<double>>> biasesVelocities;
    double velocityWeight = 0.9;
};

#endif //MATRIXBENCHMARKS_MOMENTUMOPTIMIZER_H
