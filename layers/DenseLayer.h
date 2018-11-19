//
// Created by Martin Galajda on 26/10/2018.
//
#define _GLIBCXX_USE_CXX11_ABI 0

#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "../matrix_impl/Matrix.hpp"
#include "../initializers/NormalInitializer.h"
#include "./L2Regularizer.h"
#include "ActivationFunction.h"
#include <memory>
#include <vector>
#include <memory>

enum Regularization { l2, none };

class DenseLayer {
public:
    DenseLayer(int width, int height, int batchSize, double *data, ActivationFunction activationFunction);
    DenseLayer(int width, int height, int batchSize, BaseInitializer *initializer, ActivationFunction activationFunction);

    DenseLayer & operator=(const DenseLayer&) = delete;
    DenseLayer(const DenseLayer&) = delete;

    ~DenseLayer();

    std::shared_ptr<Matrix<double>> forwardPropagate(std::shared_ptr<Matrix<double>> X);
    std::shared_ptr<Matrix<double>> activate(std::shared_ptr<Matrix<double>> &X);
    std::shared_ptr<Matrix<double>> backPropagate(std::shared_ptr<Matrix<double>> forwardDerivatives);

    void updateWeights();

    Matrix<double> &getWeightsDerivatives();
    Matrix<double> &getBiasesDerivatives() { return *this->biasesDerivatives; }
    Matrix<double> &getWeights() { return *this->weights; }
    Matrix<double> &getBiases() { return *this->biases; }

    int getWidth() { return weightsDerivatives->getNumOfCols(); }
    int getHeight() { return weightsDerivatives->getNumOfRows(); }


    void setL2Regularization(double);

    Regularization getRegularizationType() { return regularization; }
    std::shared_ptr<L2Regularizer>  getRegularizer();
protected:
    std::shared_ptr<Matrix<double>> weights;
    std::shared_ptr<Matrix<double>> inputs;
    std::shared_ptr<Matrix<double>> activatedInputs;
    std::shared_ptr<Matrix<double>> biases;

    std::shared_ptr<Matrix<double>> weightsDerivatives;
    std::shared_ptr<Matrix<double>> neuronDerivatives;
    std::shared_ptr<Matrix<double>> biasesDerivatives;
    int batchSize;

    ActivationFunction activationFunction;

    Regularization regularization = Regularization::none;
    std::shared_ptr<L2Regularizer> regularizer;
};


#endif //DENSELAYER_H
