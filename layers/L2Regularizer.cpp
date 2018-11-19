//
// Created by Martin Galajda on 30/10/2018.
//

#include "L2Regularizer.h"
#include "DenseLayer.h"

std::shared_ptr<Matrix<double>> L2Regularizer::getRegularizedWeightDerivatives() {
    auto &layerWeights = this->layer->getWeights();

    auto regularizedWeightsDerivatives = std::shared_ptr<Matrix<double>>(layerWeights * decayStrength);

    return regularizedWeightsDerivatives;
}

std::shared_ptr<Matrix<double>> L2Regularizer::getRegularizedBiasDerivatives() {
    auto &biases = this->layer->getBiases();

    auto regularizedBiasesDerivatives = std::shared_ptr<Matrix<double>>(biases * decayStrength);

    return regularizedBiasesDerivatives;
}