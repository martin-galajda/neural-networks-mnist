//
// Created by Martin Galajda on 27/10/2018.
//

#include "NormalInitializer.h"

NormalInitializer::NormalInitializer() {
    std::normal_distribution<double> distribution(0.0, 1.0);
    this->distribution = distribution;
}

const double NormalInitializer::getValue() {
    return this->distribution(generator);
}