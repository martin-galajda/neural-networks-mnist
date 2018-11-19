//
// Created by Martin Galajda on 29/10/2018.
//

#include "XavierInitializer.h"

XavierInitializer::XavierInitializer(int inputUnits, int outputUnits) {
    double deviance = std::pow(2.0 / (inputUnits + outputUnits), 0.5);
    std::normal_distribution<double> distribution(0.0, deviance);
    this->distribution = distribution;
}

const double XavierInitializer::getValue() {
    return this->distribution(generator);
}