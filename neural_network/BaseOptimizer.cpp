//
// Created by Martin Galajda on 29/10/2018.
//

#include "BaseOptimizer.h"
#include "../utilities/populate_placeholders.h"


BaseOptimizer::BaseOptimizer(
        ComputationalGraph &computationalGraph,
        std::vector<std::shared_ptr<Matrix<double>>> &instances,
        std::vector<std::shared_ptr<Matrix<double>>> &labels,
        std::vector<int> &trainIndices,
        int minibatchSize,
        double learningRate
): computationalGraph(computationalGraph), minibatchSize(minibatchSize), instances(instances), labels(labels), trainIndices(trainIndices) {

    std::random_device randomDevice;     // only used once to initialise (seed) engine
    std::mt19937 rng(randomDevice());    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> pickRandomInstance(0, trainIndices.size() -1); // guaranteed unbiased

    this->rng = rng;
    this->pickRandomTrainIndex = pickRandomInstance;

    minibatchIndices.reserve(minibatchSize);

    auto layers = computationalGraph.getLayers();
    auto firstLayer = layers.front();
    auto lastLayer = layers.back();

    inputsPlaceholder = std::shared_ptr<Matrix<double>>(new Matrix<double>(firstLayer->getWidth(), minibatchSize));
    expectedOutputsPlaceholder = std::shared_ptr<Matrix<double>>(new Matrix<double>(lastLayer->getHeight(), minibatchSize));

    this->learningRate = learningRate;
}

void BaseOptimizer::populatePlaceholdersForMinibatch() {
    populatePlaceholders(this->inputsPlaceholder, this->instances, this->minibatchIndices);
    populatePlaceholders(this->expectedOutputsPlaceholder, this->labels, this->minibatchIndices);
}