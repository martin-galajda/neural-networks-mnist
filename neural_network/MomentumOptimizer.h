//
// Created by Martin Galajda on 29/10/2018.
//

#ifndef NEURAL_NETWORKS_MOMENTUMOPTIMIZER_H
#define NEURAL_NETWORKS_MOMENTUMOPTIMIZER_H


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

    void initialize();

    virtual void train();
protected:

    std::vector<std::shared_ptr<Matrix<double>>> velocities;
    std::vector<std::shared_ptr<Matrix<double>>> biasesVelocities;
    double velocityWeight = 0.9;

  bool isInitialized = false;
};

#endif //NEURAL_NETWORKS_MOMENTUMOPTIMIZER_H
