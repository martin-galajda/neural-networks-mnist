//
// Created by Martin Galajda on 29/10/2018.
//

#include <vector>
#include "./ComputationalGraph.h"

#ifndef NEURAL_NETWORKS_BASEOPTIMIZER_H
#define NEURAL_NETWORKS_BASEOPTIMIZER_H

enum LossFunction { SoftmaxCrossEntropy };

class BaseOptimizer {
public:
    BaseOptimizer(
        ComputationalGraph &computationalGraph,
        std::vector<std::shared_ptr<Matrix<double>>> &instances,
        std::vector<std::shared_ptr<Matrix<double>>> &labels,
        std::vector<int> &trainIndices,
        int minibatchSize,
        double learningRate
    );

    void setLearningRate(double learningRate) { this->learningRate = learningRate; }
    double getLearningRate(double learningRate) { return learningRate; }

    virtual void train() = 0;

protected:
    ComputationalGraph &computationalGraph;
    std::vector<std::shared_ptr<Matrix<double>>> &instances;
    std::vector<int> &trainIndices;
    std::vector<std::shared_ptr<Matrix<double>>> &labels;

    std::mt19937 rng;
    std::uniform_int_distribution<int> pickRandomTrainIndex;
    std::vector<int> minibatchIndices;

    std::shared_ptr<Matrix<double>> inputsPlaceholder = nullptr;
    std::shared_ptr<Matrix<double>> expectedOutputsPlaceholder = nullptr;

    int minibatchSize;
    double learningRate;
    double movingAverageAcc = 0;
    double movingAverageAccCount = 0;

    void populatePlaceholdersForMinibatch();
};


#endif //NEURAL_NETWORKS_BASEOPTIMIZER_H
