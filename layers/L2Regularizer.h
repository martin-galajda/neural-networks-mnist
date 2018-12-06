//
// Created by Martin Galajda on 30/10/2018.
//


class BaseLayer;

#include <memory>
#include "../matrix_impl/Matrix.hpp"

#ifndef L2REGULARIZER_H
#define L2REGULARIZER_H

class L2Regularizer {
public :
    L2Regularizer(BaseLayer *layer, double decayStrength): layer(layer), decayStrength(decayStrength) {};

    std::shared_ptr<Matrix<double>> getRegularizedWeightDerivatives();
    std::shared_ptr<Matrix<double>> getRegularizedBiasDerivatives();
protected:
    BaseLayer *layer;
    double decayStrength;
};


#endif //L2REGULARIZER_H
