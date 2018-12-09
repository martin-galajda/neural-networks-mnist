//
// Created by Martin Galajda on 01/11/2018.
//

#include "AdamOptimizer.h"
#include "../initializers/ZeroInitializer.h"
#include "../utilities/report_accuracy.h"


void AdamOptimizer::train() {
    minibatchIndices.clear();
    for (auto i = 0; i < minibatchSize; i++) {
        minibatchIndices.push_back(trainIndices[pickRandomTrainIndex(rng)]);
    }

    populatePlaceholdersForMinibatch();

    auto s = computationalGraph.forwardPass(inputsPlaceholder);

    auto batchAccuracy = reportAccuracy(s, expectedOutputsPlaceholder);

    auto newMovingAccuracy = (batchAccuracy + (this->movingAverageAcc * this->movingAverageAccCount)) / (this->movingAverageAccCount + 1);
    this->movingAverageAccCount += 1;
    this->movingAverageAcc = newMovingAccuracy;

    auto processedCount = this->movingAverageAccCount * this->minibatchSize;

    if ((int) this->movingAverageAccCount % this->reportEveryBatch == 0) {
        std::cout
            << "\r"
            << "Processed: (" << processedCount
            << "/" << this->epochSize << "). "
            << "Moving cumulative mean accuracy: " << this->movingAverageAcc << std::flush;
    }

    if ((int) processedCount > epochSize) {
        this->movingAverageAccCount = 1;
        this->movingAverageAcc = batchAccuracy;
    }


    // we can mutate softmax outputs as we are using it just for computing derivatives...
    auto lossDerivatives = MatrixDoubleSharedPtr(*s - (*expectedOutputsPlaceholder));

    computationalGraph.backwardPass(lossDerivatives, numOfThreads);

    timestep += 1;

    auto layers = computationalGraph.getLayers();

    auto layerWeightIndex = 0;
    auto layerBiasIndex = 0;

    if (!this->isInitialized) {
        this->initialize();
    }

    for (auto layerIt = layers.begin(); layerIt != layers.end(); layerIt++) {
        auto learningRateAtCurrentTimestamp = learningRate * (std::sqrt(1.0 - std::pow(beta2, timestep))) * (1.0 - std::pow(beta1, timestep));

        if ((*layerIt)->hasWeights()) {
            auto &weightsDerivatives = (*layerIt)->getWeightsDerivatives();
            auto weightDerivativesSquared = std::shared_ptr<Matrix<double>>(weightsDerivatives.pow(2));
            auto &layerGradientAverages = this->gradientAverages[layerWeightIndex];
            auto &layerSquaredGradientAverages = this->squaredGradientAverages[layerWeightIndex];

            *layerGradientAverages *= beta1;

            auto adjustedWeighDerivatives = std::shared_ptr<Matrix<double>>(weightsDerivatives * (1.0 - beta1));
            *layerGradientAverages += *adjustedWeighDerivatives;
            *layerSquaredGradientAverages *= beta2;

            auto adjustedSquaredDerivatives = std::shared_ptr<Matrix<double>>((*weightDerivativesSquared) * ((1.0 - beta2) / minibatchSize));
            *layerSquaredGradientAverages += *adjustedSquaredDerivatives;

            auto squareRootsSecondMoments = std::shared_ptr<Matrix<double>>(layerSquaredGradientAverages->sqrt());
            *squareRootsSecondMoments += epsilonCorrection;
            auto weightUpdates = std::shared_ptr<Matrix<double>>((*layerGradientAverages) / (*squareRootsSecondMoments));
            *weightUpdates *= learningRateAtCurrentTimestamp;
            weightsDerivatives.copyElementsFrom(*weightUpdates);

            layerWeightIndex++;
        }

        if ((*layerIt)->hasBiases()) {
            auto &biasesDerivatives = (*layerIt)->getBiasesDerivatives();


            auto biasesDerivativesSquared = std::shared_ptr<Matrix<double>>(biasesDerivatives.pow(2));

            auto &layerBiasGradientAverages = this->gradientAveragesBiases[layerBiasIndex];
            auto &layerBiasSquaredGradientAverages = this->squaredGradientAveragesBiases[layerBiasIndex];

            *layerBiasGradientAverages *= beta1;
            auto adjustedBiasDerivatives = std::shared_ptr<Matrix<double>>(biasesDerivatives * ((1.0 - beta1) / minibatchSize));
            *layerBiasGradientAverages += *adjustedBiasDerivatives;
            *layerBiasSquaredGradientAverages *= beta2;

            auto adjustedSquaredBiasDerivatives = std::shared_ptr<Matrix<double>>((*biasesDerivativesSquared) * (1.0 - beta2));
            *layerBiasSquaredGradientAverages += *adjustedSquaredBiasDerivatives;
            auto squareRootsBiasesSecondMoments = std::shared_ptr<Matrix<double>>(layerBiasSquaredGradientAverages->sqrt());
            *squareRootsBiasesSecondMoments += epsilonCorrection;
            auto biasesUpdates = std::shared_ptr<Matrix<double>>((*layerBiasGradientAverages) / (*squareRootsBiasesSecondMoments));
            *biasesUpdates *= learningRateAtCurrentTimestamp;
            biasesDerivatives.copyElementsFrom(*biasesUpdates);

            layerBiasIndex++;
        }



//        if ((*layerIt)->getRegularizationType() == Regularization::l2) {
//           auto regularizer = (*layerIt)->getRegularizer();
//           auto regularizedWeightDerivatives = regularizer->getRegularizedWeightDerivatives();
//           *regularizedWeightDerivatives *= (learningRate / minibatchSize);
//           weightsDerivatives += *regularizedWeightDerivatives;
//
//// TODO: Do we really ignore biases in L2 regularization?
////         auto regularizedBiasDerivatives = regularizer->getRegularizedBiasDerivatives();
////         *regularizedBiasDerivatives *= (learningRate / minibatchSize);
////         biasesDerivatives += (*regularizedBiasDerivatives);
//        }

    }

    computationalGraph.learn();


}

AdamOptimizer::AdamOptimizer(ComputationalGraph &computationalGraph,
    std::vector<std::shared_ptr<Matrix<double>>> &instances,
    std::vector<std::shared_ptr<Matrix<double>>> &labels,
    std::vector<int> &trainIndices,
    int minibatchSize,
    double learningRate,
    int numOfThreads,
    int epochSize,
    int reportEveryBatch
    ):
        BaseOptimizer(computationalGraph, instances, labels, trainIndices, minibatchSize, learningRate, numOfThreads),
        epochSize(epochSize),
        reportEveryBatch(reportEveryBatch)
    {
}

void AdamOptimizer::initialize() {
    auto layers = computationalGraph.getLayers();

    ZeroInitializer zeroInitializer;

    for (auto layerIt = layers.begin(); layerIt != layers.end(); layerIt++) {
        auto &layer = *layerIt;
        if (layer->hasWeights()) {
            auto batchSize = (*layerIt)->getBatchSize();
            auto layerWidth = (*layerIt)->getWidth();
            auto layerHeight = (*layerIt)->getHeight();
            auto layerDepth = (*layerIt)->getDepth();

            auto layerSquaredGradientAverages = std::shared_ptr<Matrix<double>>(new Matrix<double>(layerHeight, layerWidth, layerDepth, batchSize, &zeroInitializer));
            auto layerGradientAverages = std::shared_ptr<Matrix<double>>(new Matrix<double>(layerHeight, layerWidth, layerDepth, batchSize, &zeroInitializer));

            this->gradientAverages.push_back(layerGradientAverages);
            this->squaredGradientAverages.push_back(layerSquaredGradientAverages);
        }

        if (layer->hasBiases()) {
            auto layerHeight = (*layerIt)->getHeight();

            auto layerSquaredGradientAveragesBiases = std::shared_ptr<Matrix<double>>(new Matrix<double>(layerHeight, 1, &zeroInitializer));
            auto layerGradientAveragesBiases = std::shared_ptr<Matrix<double>>(new Matrix<double>(layerHeight, 1, &zeroInitializer));

            this->squaredGradientAveragesBiases.push_back(layerSquaredGradientAveragesBiases);

            this->gradientAveragesBiases.push_back(layerGradientAveragesBiases);
        }
    }

    this->isInitialized = true;
}
