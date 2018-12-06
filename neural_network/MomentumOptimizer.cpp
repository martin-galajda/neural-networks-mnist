//
// Created by Martin Galajda on 29/10/2018.
//

#include "MomentumOptimizer.h"
#include "../initializers/ZeroInitializer.h"
#include "../utilities/report_accuracy.h"


void MomentumOptimizer::train() {
    minibatchIndices.clear();
    for (auto i = 0; i < minibatchSize; i++) {
        minibatchIndices.push_back(trainIndices[pickRandomTrainIndex(rng)]);
    }

    populatePlaceholdersForMinibatch();

    auto s = computationalGraph.forwardPass(inputsPlaceholder);

    double batchAccuracy = reportAccuracy(s, expectedOutputsPlaceholder);

    auto lossDerivatives = MatrixDoubleSharedPtr(*s - (*expectedOutputsPlaceholder));

    computationalGraph.backwardPass(lossDerivatives);

    auto &layers = computationalGraph.getLayers();

    auto layerWeightIndex = 0;
    auto layerBiasIndex = 0;

    auto currLayerIdx = layers.size() - 1;
    for (auto layerIt = layers.begin(); layerIt != layers.end(); layerIt++) {
        auto &layer = *layerIt;

        if (layer->hasWeights()) {
            auto &weightsDerivatives = layer->getWeightsDerivatives();

            // update weight derivatives
            auto &layerVelocities = this->velocities[layerWeightIndex];
            *layerVelocities *= velocityWeight;
            weightsDerivatives *= (learningRate / minibatchSize);
            *layerVelocities += weightsDerivatives;

            weightsDerivatives.copyElementsFrom(*layerVelocities);

//            if (layer->getRegularizationType() == Regularization::l2) {
//                auto regularizer = layer->getRegularizer();
//                auto regularizedWeightDerivatives = regularizer->getRegularizedWeightDerivatives();
//                *regularizedWeightDerivatives *= (learningRate / minibatchSize);
//                weightsDerivatives += *regularizedWeightDerivatives;

//            auto regularizedBiasDerivatives = regularizer->getRegularizedBiasDerivatives();
//            *regularizedBiasDerivatives *= (learningRate / minibatchSize);
//            biasesDerivatives += (*regularizedBiasDerivatives);
//            }

            layerWeightIndex += 1;
        }


        // update bias derivatives
        if (layer->hasBiases()) {
            auto &biasesDerivatives = layer->getBiasesDerivatives();
            auto &biasesVelocities = this->biasesVelocities[layerBiasIndex];
            *biasesVelocities *= velocityWeight;
            biasesDerivatives *= (learningRate / minibatchSize);
            *biasesVelocities += biasesDerivatives;
            biasesDerivatives.copyElementsFrom(*biasesVelocities);

//            if ((*layerIt)->getRegularizationType() == Regularization::l2) {
//            auto regularizedBiasDerivatives = regularizer->getRegularizedBiasDerivatives();
//            *regularizedBiasDerivatives *= (learningRate / minibatchSize);
//            biasesDerivatives += (*regularizedBiasDerivatives);
//            }

            layerBiasIndex += 1;
        }

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
        if ((*layerIt)->hasWeights()) {
            auto layerWidth = (*layerIt)->getWidth();
            auto layerHeight = (*layerIt)->getHeight();
            auto layerDepth = (*layerIt)->getDepth();
            auto layerBatchSize = (*layerIt)->getBatchSize();

            auto layerVelocities = std::shared_ptr<Matrix<double>>(new Matrix<double>(layerHeight, layerWidth, layerDepth, layerBatchSize, &zeroInitializer));

            this->velocities.push_back(std::move(layerVelocities));
        }

        if ((*layerIt)->hasBiases()) {
            auto layerHeight = (*layerIt)->getHeight();
            auto biasesVelocities = std::shared_ptr<Matrix<double>>(new Matrix<double>(layerHeight, 1, &zeroInitializer));
            this->biasesVelocities.push_back(std::move(biasesVelocities));
        }
    }
}
