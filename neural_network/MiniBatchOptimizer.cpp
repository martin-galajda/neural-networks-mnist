//
// Created by Martin Galajda on 29/10/2018.
//

#include "MiniBatchOptimizer.h"

void reportAccuracyMinibatch(MatrixDoubleSharedPtr predicted, MatrixDoubleSharedPtr expected) {
    auto predictedValues = predicted->argMaxByRow();
    auto expectedValues = expected->argMaxByRow();

    if (predictedValues->getNumOfRows() != expectedValues->getNumOfRows()) {
        throw std::invalid_argument("Mismatched no of examples when computing accuracy.");
    }

    auto matchedPredictions = 0;
    for (auto row = 0; row < predictedValues->getNumOfRows(); row++) {
        if ((*predictedValues)[row][0] == (*expectedValues)[row][0]) {
            matchedPredictions++;
        }
    }


    std::cout << "Accuracy: " << (matchedPredictions / 1.0) / predictedValues->getNumOfRows()
              << std::endl;
}


void MiniBatchOptimizer::train() {
    minibatchIndices.clear();
    for (auto i = 0; i < minibatchSize; i++) {
        minibatchIndices.push_back(trainIndices[pickRandomTrainIndex(rng)]);
    }

    populatePlaceholdersForMinibatch();

    auto s = computationalGraph.forwardPass(inputsPlaceholder);


    // we can mutate softmax outputs as we are using it just for computing derivatives...
    // But do we really care that much?
    // *s -= (*expectedOutputsPlaceholder);


    reportAccuracyMinibatch(s, expectedOutputsPlaceholder);

    auto lossDerivatives = std::shared_ptr<Matrix<double>>(*s - *expectedOutputsPlaceholder);

//    (*lossDerivatives) /= minibatchSize;

    computationalGraph.backwardPass(lossDerivatives);

    auto layers = computationalGraph.getLayers();
    int layerIndex = layers.size() - 1;

    for (auto layerIt = layers.rbegin(); layerIt != layers.rend(); layerIt++) {

        if ((*layerIt)->hasWeights()) {
            auto &weightsDerivatives = (*layerIt)->getWeightsDerivatives();
            weightsDerivatives *= (learningRate / minibatchSize);
//            weightsDerivatives /= minibatchSize;
        }

        if ((*layerIt)->hasBiases()) {
            auto &biasesDerivatives = (*layerIt)->getBiasesDerivatives();
            biasesDerivatives *= (learningRate / minibatchSize);
//            biasesDerivatives /= minibatchSize;
        }
    }

    computationalGraph.learn();
}
