//
// Created by Martin Galajda on 29/10/2018.
//

#include "MiniBatchOptimizer.h"

void MiniBatchOptimizer::train() {
    minibatchIndices.clear();
    for (auto i = 0; i < minibatchSize; i++) {
        minibatchIndices.push_back(trainIndices[pickRandomTrainIndex(rng)]);
    }

    populatePlaceholdersForMinibatch();

    auto s = computationalGraph.forwardPass(inputsPlaceholder);

    // we can mutate softmax outputs as we are using it just for computing derivatives...
//    *s -= (*expectedOutputsPlaceholder);

    auto lossDerivatives = std::shared_ptr<Matrix<double>>(*s - *expectedOutputsPlaceholder);

    (*lossDerivatives) /= minibatchSize;

    computationalGraph.backwardPass(lossDerivatives);

    auto layers = computationalGraph.getLayers();
    int layerIndex = layers.size() - 1;

    for (auto layerIt = layers.rbegin(); layerIt != layers.rend(); layerIt++) {
        auto &weightsDerivatives = (*layerIt)->getWeightsDerivatives();
        auto &biasesDerivatives = (*layerIt)->getBiasesDerivatives();

        weightsDerivatives *= learningRate;
        biasesDerivatives *= learningRate;
    }

    computationalGraph.learn();
}
