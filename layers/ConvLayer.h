//
// Created by Martin Galajda on 27/11/2018.
//

#ifndef MATRIXBENCHMARKS_CONVLAYER_H
#define MATRIXBENCHMARKS_CONVLAYER_H

#include "BaseLayer.h"

class ConvLayer: public BaseLayer {

public:
    ConvLayer(
            int kernelWidth,
            int kernelHeight,
            int batchSize,
            int inputDepth,
            int numberOfFilters,
            BaseInitializer *initializer,
            ActivationFunction activationFunction,
            int stride,
            std::string name = "ConvLayer"
    );

    virtual std::shared_ptr<Matrix<double>> forwardPropagate(std::shared_ptr<Matrix<double>> X);
    virtual std::shared_ptr<Matrix<double>> activate(std::shared_ptr<Matrix<double>> &X);
    virtual std::shared_ptr<Matrix<double>> backPropagate(std::shared_ptr<Matrix<double>> forwardDerivatives);

    virtual bool hasBiases();
    virtual bool hasWeights();

protected:
    int stride;
    int kernelWidth;
    int kernelHeight;
    int inputDepth;
    int numberOfFilters;
};


#endif //MATRIXBENCHMARKS_CONVLAYER_H
