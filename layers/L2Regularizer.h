//
// Created by Martin Galajda on 30/10/2018.
//

#define _GLIBCXX_USE_CXX11_ABI 0

class DenseLayer;

#include <memory>
#include "../matrix_impl/Matrix.hpp"

#ifndef L2REGULARIZER_H
#define L2REGULARIZER_H

class L2Regularizer {
public :
    L2Regularizer(DenseLayer *layer, double decayStrength): layer(layer), decayStrength(decayStrength) {};

    std::shared_ptr<Matrix<double>> getRegularizedWeightDerivatives();
    std::shared_ptr<Matrix<double>> getRegularizedBiasDerivatives();
protected:
    DenseLayer *layer;
    double decayStrength;
};


#endif //L2REGULARIZER_H
