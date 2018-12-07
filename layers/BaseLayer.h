//
// Created by Martin Galajda on 26/10/2018.
//

#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "../matrix_impl/Matrix.hpp"
#include "../initializers/NormalInitializer.h"
#include "ActivationFunction.h"
#include <memory>
#include <vector>
#include <memory>
#include "./L2Regularizer.h"

enum Regularization { l2, none };

class BaseLayer {
public:
    BaseLayer(std::string name = "BaseLayer"): name(name) {}
    BaseLayer(int batchSize, std::string name = "BaseLayer"): batchSize(batchSize), name(name) {}
    BaseLayer & operator=(const BaseLayer&) = delete;
    BaseLayer(const BaseLayer&) = delete;

    virtual std::shared_ptr<Matrix<double>> forwardPropagate(std::shared_ptr<Matrix<double>> X) = 0;
    virtual std::shared_ptr<Matrix<double>> activate(std::shared_ptr<Matrix<double>> &X) = 0;
    virtual std::shared_ptr<Matrix<double>> backPropagate(std::shared_ptr<Matrix<double>> forwardDerivatives) = 0;

    void updateWeights();

    inline Matrix<double> &getWeightsDerivatives() { return *this->weightsDerivatives; };
    inline Matrix<double> &getBiasesDerivatives() { return *this->biasesDerivatives; }
    inline Matrix<double> &getWeights() { return *this->weights; }
    inline Matrix<double> &getBiases() { return *this->biases; }

    virtual bool hasBiases() = 0;
    virtual bool hasWeights() = 0;

    inline int getWidth() { return weights->getNumOfCols(); }
    inline int getHeight() { return weights->getNumOfRows(); }
    inline int getDepth() { return weights->getDepth(); }
    inline int getBatchSize() { return weights->getBatchSize(); }

    virtual int getLayerOutputDepth() { return 1; }

    void setL2Regularization(double decayStrength);

    Regularization getRegularizationType() { return regularization; }
    std::shared_ptr<L2Regularizer> getRegularizer();
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

    std::string name;
};


#endif //DENSELAYER_H
