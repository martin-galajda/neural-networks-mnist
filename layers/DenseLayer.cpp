//
// Created by Martin Galajda on 26/10/2018.
//

#include "DenseLayer.h"
#include <vector>
#include "../matrix_impl/Matrix.hpp"
#include "../initializers/ZeroInitializer.h"
#include "./L2Regularizer.h"
#include <sstream>
#include <stdexcept>
#include <iomanip>

DenseLayer::~DenseLayer() {}

DenseLayer::DenseLayer(int width, int height, int batchSize, BaseInitializer *initializer, ActivationFunction activationFunction) {
    this->weights = std::shared_ptr<Matrix<double>>(new Matrix<double>(height, width, initializer));
    this->inputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(width, batchSize));
    this->activatedInputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(height, batchSize));
    this->activationFunction = activationFunction;
    this->batchSize = batchSize;

    auto zeroInitializer = new ZeroInitializer();

    this->weightsDerivatives = std::shared_ptr<Matrix<double> >(new Matrix<double>(height, width));
    this->neuronDerivatives = std::shared_ptr<Matrix<double> >(new Matrix<double>(width, batchSize));
    this->biases = std::shared_ptr<Matrix<double> >(new Matrix<double>(height, 1, zeroInitializer));
    this->biasesDerivatives = std::shared_ptr<Matrix<double> >(new Matrix<double>(height, 1));

    free(zeroInitializer);
}

DenseLayer::DenseLayer(int width, int height, int batchSize, double *data, ActivationFunction activationFunction) {
    this->weights = std::shared_ptr<Matrix<double>>(new Matrix<double>(data, height, width));
    this->activationFunction = activationFunction;
    this->batchSize = batchSize;
    this->inputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(width, batchSize));
    this->activatedInputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(width, batchSize));

    this->weightsDerivatives = std::shared_ptr<Matrix<double> >(new Matrix<double>(height, width));
    this->neuronDerivatives = std::shared_ptr<Matrix<double> >(new Matrix<double>(width, batchSize));
}

std::shared_ptr<Matrix<double>> DenseLayer::forwardPropagate(std::shared_ptr<Matrix<double>> X){
    this->inputs = X;

    this->inputs->copyElementsFrom(*X);
    auto A = std::shared_ptr<Matrix<double>>((*this->weights) * (*X));

    for (auto i = 0; i < A->getNumOfRows(); i++) {
        for (auto j = 0; j < A ->getNumOfCols(); j++) {
            (*A)[i][j] += (*biases)[i][0];
        }
    }

    auto activatedInputs = this->activate(A);

    this->activatedInputs->copyElementsFrom(*activatedInputs);

    return activatedInputs;
}


std::shared_ptr<Matrix<double>> DenseLayer::backPropagate(std::shared_ptr<Matrix<double>> forwardDerivativesPtr) {
    Matrix<double> &forwardDerivatives = *forwardDerivativesPtr;

    if (forwardDerivatives.getNumOfCols() != this->batchSize) {
        std::stringstream errorString;
        errorString << "Got invalid number of columns for forward derivatives. Expected:  "
            << this->batchSize
            << " but I got "
            << forwardDerivatives.getNumOfCols()
            << std::endl;

        throw std::invalid_argument(errorString.str());
    }

    Matrix<double> &weightDerivatives = *(this->weightsDerivatives);
    Matrix<double> &neuronDerivatives = *(this->neuronDerivatives);

    weightDerivatives.setAllElementsZero(0.0);
    biasesDerivatives->setAllElementsZero(0.0);
    neuronDerivatives.setAllElementsZero(0.0);

    // each column contains derivatives for one training example
    for (int instanceIndex = 0; instanceIndex < forwardDerivatives.getNumOfCols(); instanceIndex++) {
        for (int neuronIndex = 0; neuronIndex < (*inputs).getNumOfRows(); neuronIndex++) {
            for (int forwardDerivativeIndex = 0; forwardDerivativeIndex < forwardDerivatives.getNumOfRows(); forwardDerivativeIndex++) {

                auto currForwardDerivativeValue = forwardDerivatives[forwardDerivativeIndex][instanceIndex];
                auto currActivatedInputValue = (*activatedInputs)[forwardDerivativeIndex][instanceIndex];

                if (this->activationFunction == ActivationFunction::relu) {
                    double reluDerivative = 0.0;
                    if (currActivatedInputValue > 0) {
                        reluDerivative = 1.0;
                    }
                    // we need to include relu derivative

                    (weightDerivatives)[forwardDerivativeIndex][neuronIndex] += (*inputs)[neuronIndex][instanceIndex] * reluDerivative * currForwardDerivativeValue;

                    (neuronDerivatives)[neuronIndex][instanceIndex] +=
                            (*weights)[forwardDerivativeIndex][neuronIndex] * reluDerivative * currForwardDerivativeValue;

                    (*biasesDerivatives)[forwardDerivativeIndex][0] += (currForwardDerivativeValue * reluDerivative) / (*inputs).getNumOfRows();
                } else {

                    // softmax derivative already has derivative of activation function passed
                    (weightDerivatives)[forwardDerivativeIndex][neuronIndex] += (*inputs)[neuronIndex][instanceIndex] * currForwardDerivativeValue;
                    (neuronDerivatives)[neuronIndex][instanceIndex] += (*weights)[forwardDerivativeIndex][neuronIndex] * currForwardDerivativeValue;
                    (*biasesDerivatives)[forwardDerivativeIndex][0] += (currForwardDerivativeValue) / (*inputs).getNumOfRows();
                }

            }
        }
    };


    return this->neuronDerivatives;
}

void DenseLayer::updateWeights() {
    *this->weights -= *weightsDerivatives;
    *this->biases -= *biasesDerivatives;
}

std::shared_ptr<Matrix<double>> DenseLayer::activate(std::shared_ptr<Matrix<double>> &A) {
    if (this->activationFunction == ActivationFunction::softmax) {
        return std::shared_ptr<Matrix<double>>(A->softmax());
    } else if (this->activationFunction == ActivationFunction::relu) {
        A->reluInPlace();
        return A;
    }

    return nullptr;
}

Matrix<double>& DenseLayer::getWeightsDerivatives() {
    return *this->weightsDerivatives;
}

void DenseLayer::setL2Regularization(double decayStrength) {
    this->regularization = Regularization :: l2;
    this->regularizer = std::shared_ptr<L2Regularizer>(new L2Regularizer(this, decayStrength));
}

std::shared_ptr<L2Regularizer> DenseLayer::getRegularizer() {
    return this->regularizer;
}