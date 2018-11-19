//
// Created by Martin Galajda on 27/10/2018.
//
#define _GLIBCXX_USE_CXX11_ABI 0


#include <random>
#include "BaseInitializer.h"

#ifndef MATRIXBENCHMARKS_NORMALINITIALIZER_H
#define MATRIXBENCHMARKS_NORMALINITIALIZER_H


class NormalInitializer: public BaseInitializer {
public:
    NormalInitializer();

    const double getValue();
protected:

    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
};


#endif //MATRIXBENCHMARKS_NORMALINITIALIZER_H
