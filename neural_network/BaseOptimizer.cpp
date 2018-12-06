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

    if (computationalGraph.getInputSizeCols() > 1) {
        // 2D
        inputsPlaceholder = std::shared_ptr<Matrix<double>>(new Matrix<double>(computationalGraph.getInputSizeRows(), computationalGraph.getInputSizeCols(), 1, minibatchSize));
        expectedOutputsPlaceholder = std::shared_ptr<Matrix<double>>(new Matrix<double>(computationalGraph.getOutputSize(), 1, 1, minibatchSize));
    } else {
        auto layers = computationalGraph.getLayers();
        auto firstLayer = layers.front();
        auto lastLayer = layers.back();

        inputsPlaceholder = std::shared_ptr<Matrix<double>>(new Matrix<double>(firstLayer->getWidth(), minibatchSize));
        expectedOutputsPlaceholder = std::shared_ptr<Matrix<double>>(new Matrix<double>(lastLayer->getHeight(), 1, 1, minibatchSize));
    }

    this->learningRate = learningRate;
    this->movingAverageAcc = 0;
}

void BaseOptimizer::populatePlaceholdersForMinibatch() {
    if (this->inputsPlaceholder->getBatchSize() > 1) {
        // 2D
        populatePlaceholders2D(this->inputsPlaceholder, this->instances, this->minibatchIndices);
    } else {
        populatePlaceholders(this->inputsPlaceholder, this->instances, this->minibatchIndices);
    }
    populatePlaceholders(this->expectedOutputsPlaceholder, this->labels, this->minibatchIndices);
}