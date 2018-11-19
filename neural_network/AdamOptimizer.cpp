//
// Created by Martin Galajda on 01/11/2018.
//

#include "AdamOptimizer.h"
#include "../initializers/ZeroInitializer.h"

void AdamOptimizer::train() {
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


    timestep += 1;

    auto layers = computationalGraph.getLayers();
    int layerIndex = layers.size() - 1;
    for (auto layerIt = layers.rbegin(); layerIt != layers.rend(); layerIt++) {
        auto &weightsDerivatives = (*layerIt)->getWeightsDerivatives();
        auto &biasesDerivatives = (*layerIt)->getBiasesDerivatives();

        auto weightDerivativesSquared = std::shared_ptr<Matrix<double>>(weightsDerivatives.pow(2));

        // update weight derivatives
        auto &layerGradientAverages = this->gradientAverages[layerIndex];
        auto &layerSquaredGradientAverages = this->squaredGradientAverages[layerIndex];
        auto learningRateAtCurrentTimestamp = learningRate * (std::sqrt(1.0 - std::pow(beta2, timestep))) * (1.0 - std::pow(beta1, timestep));

        *layerGradientAverages *= beta1;

        auto adjustedWeighDerivatives = std::shared_ptr<Matrix<double>>(weightsDerivatives * (1.0 - beta1));
        *layerGradientAverages += *adjustedWeighDerivatives;
        *layerSquaredGradientAverages *= beta2;

        auto adjustedSquaredDerivatives = std::shared_ptr<Matrix<double>>((*weightDerivativesSquared) * (1.0 - beta2));
        *layerSquaredGradientAverages += *adjustedSquaredDerivatives;

        auto squareRootsSecondMoments = std::shared_ptr<Matrix<double>>(layerSquaredGradientAverages->sqrt());
        *squareRootsSecondMoments += epsilonCorrection;
        auto weightUpdates = std::shared_ptr<Matrix<double>>((*layerGradientAverages) / (*squareRootsSecondMoments));
        *weightUpdates *= learningRateAtCurrentTimestamp;
        weightsDerivatives.copyElementsFrom(*weightUpdates);

        auto biasesDerivativesSquared = std::shared_ptr<Matrix<double>>(biasesDerivatives.pow(2));

        auto &layerBiasGradientAverages = this->gradientAveragesBiases[layerIndex];
        auto &layerBiasSquaredGradientAverages = this->squaredGradientAveragesBiases[layerIndex];

        *layerBiasGradientAverages *= beta1;
        auto adjustedBiasDerivatives = std::shared_ptr<Matrix<double>>(biasesDerivatives * (1.0 - beta1));
        *layerBiasGradientAverages += *adjustedBiasDerivatives;
        *layerBiasSquaredGradientAverages *= beta2;

        auto adjustedSquaredBiasDerivatives = std::shared_ptr<Matrix<double>>((*biasesDerivativesSquared) * (1.0 - beta2));
        *layerBiasSquaredGradientAverages += *adjustedSquaredBiasDerivatives;
        auto squareRootsBiasesSecondMoments = std::shared_ptr<Matrix<double>>(layerBiasSquaredGradientAverages->sqrt());
        *squareRootsBiasesSecondMoments += epsilonCorrection;
        auto biasesUpdates = std::shared_ptr<Matrix<double>>((*layerBiasGradientAverages) / (*squareRootsBiasesSecondMoments));
        *biasesUpdates *= learningRateAtCurrentTimestamp;
        biasesDerivatives.copyElementsFrom(*biasesUpdates);

        if ((*layerIt)->getRegularizationType() == Regularization::l2) {
//            auto regularizer = (*layerIt)->getRegularizer();
//            auto regularizedWeightDerivatives = regularizer->getRegularizedWeightDerivatives();
//            *regularizedWeightDerivatives *= (learningRate / minibatchSize);
//            weightsDerivatives += *regularizedWeightDerivatives;

//            auto regularizedBiasDerivatives = regularizer->getRegularizedBiasDerivatives();
//            *regularizedBiasDerivatives *= (learningRate / minibatchSize);
//            biasesDerivatives += (*regularizedBiasDerivatives);
        }

        layerIndex--;
    }

    computationalGraph.learn();


}

AdamOptimizer::AdamOptimizer(ComputationalGraph &computationalGraph,
                                     std::vector<std::shared_ptr<Matrix<double>>> &instances,
                                     std::vector<std::shared_ptr<Matrix<double>>> &labels,
                                     std::vector<int> &trainIndices, int minibatchSize, double learningRate):

        BaseOptimizer(computationalGraph, instances, labels, trainIndices, minibatchSize, learningRate) {
    auto layers = computationalGraph.getLayers();

    ZeroInitializer zeroInitializer;

    for (auto layerIt = layers.begin(); layerIt != layers.end(); layerIt++) {
        auto layerWidth = (*layerIt)->getWidth();
        auto layerHeight = (*layerIt)->getHeight();

        auto layerSquaredGradientAverages = std::shared_ptr<Matrix<double>>(new Matrix<double>(layerHeight, layerWidth, &zeroInitializer));
        auto layerGradientAverages = std::shared_ptr<Matrix<double>>(new Matrix<double>(layerHeight, layerWidth, &zeroInitializer));

        auto layerSquaredGradientAveragesBiases = std::shared_ptr<Matrix<double>>(new Matrix<double>(layerHeight, 1, &zeroInitializer));
        auto layerGradientAveragesBiases = std::shared_ptr<Matrix<double>>(new Matrix<double>(layerHeight, 1, &zeroInitializer));

        this->squaredGradientAveragesBiases.push_back(layerSquaredGradientAveragesBiases);
        this->squaredGradientAverages.push_back(layerSquaredGradientAverages);

        this->gradientAverages.push_back(layerGradientAverages);
        this->gradientAveragesBiases.push_back(layerGradientAveragesBiases);
    }
}
