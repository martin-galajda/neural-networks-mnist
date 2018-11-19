//
// Created by Martin Galajda on 29/10/2018.
//

#include "MomentumOptimizer.h"
#include "../initializers/ZeroInitializer.h"


void MomentumOptimizer::train() {
    minibatchIndices.clear();
    for (auto i = 0; i < minibatchSize; i++) {
        minibatchIndices.push_back(trainIndices[pickRandomTrainIndex(rng)]);
    }

    populatePlaceholdersForMinibatch();

    auto s = computationalGraph.forwardPass(inputsPlaceholder);

    // we can mutate softmax outputs as we are using it just for computing derivatives...
    *s -= (*expectedOutputsPlaceholder);
    auto &lossDerivatives = s;

    (*lossDerivatives) /= minibatchSize;

    computationalGraph.backwardPass(lossDerivatives);

    auto &layers = computationalGraph.getLayers();
    int layerIndex = layers.size() - 1;
    for (auto layerIt = layers.rbegin(); layerIt != layers.rend(); layerIt++) {

        auto &weightsDerivatives = (*layerIt)->getWeightsDerivatives();
        auto &biasesDerivatives = (*layerIt)->getBiasesDerivatives();

        // update weight derivatives
        auto layerVelocities = this->velocities[layerIndex];
        *layerVelocities *= velocityWeight;
        weightsDerivatives *= learningRate;
        *layerVelocities += weightsDerivatives;


        // update bias derivatives
        auto biasesVelocities = this->biasesVelocities[layerIndex];
        *biasesVelocities *= velocityWeight;
        biasesDerivatives *= learningRate;
        *biasesVelocities += biasesDerivatives;

        weightsDerivatives.copyElementsFrom(*layerVelocities);
        biasesDerivatives.copyElementsFrom(*biasesVelocities);

        if ((*layerIt)->getRegularizationType() == Regularization::l2) {
            auto regularizer = (*layerIt)->getRegularizer();
            auto regularizedWeightDerivatives = regularizer->getRegularizedWeightDerivatives();
            *regularizedWeightDerivatives *= (learningRate / minibatchSize);
            weightsDerivatives += *regularizedWeightDerivatives;

//            auto regularizedBiasDerivatives = regularizer->getRegularizedBiasDerivatives();
//            *regularizedBiasDerivatives *= (learningRate / minibatchSize);
//            biasesDerivatives += (*regularizedBiasDerivatives);
        }

        layerIndex--;
    }

    computationalGraph.learn();
}

MomentumOptimizer::MomentumOptimizer(ComputationalGraph &computationalGraph,
     std::vector<std::shared_ptr<Matrix<double>>> &instances,
     std::vector<std::shared_ptr<Matrix<double>>> &labels,
     std::vector<int> &trainIndices, int minibatchSize, double learningRate):

     BaseOptimizer(computationalGraph, instances, labels, trainIndices, minibatchSize, learningRate) {
    auto &layers = computationalGraph.getLayers();

    ZeroInitializer zeroInitializer;


    for (auto layerIt = layers.begin(); layerIt != layers.end(); layerIt++) {
        auto layerWidth = (*layerIt)->getWidth();
        auto layerHeight = (*layerIt)->getHeight();

        auto layerVelocities = std::shared_ptr<Matrix<double>>(new Matrix<double>(layerHeight, layerWidth, &zeroInitializer));
        auto biasesVelocities = std::shared_ptr<Matrix<double>>(new Matrix<double>(layerHeight, 1, &zeroInitializer));

        this->velocities.push_back(layerVelocities);
        this->biasesVelocities.push_back(biasesVelocities);
    }
}
