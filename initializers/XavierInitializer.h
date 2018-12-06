//
// Created by Martin Galajda on 29/10/2018.
//

#include <random>
#include "BaseInitializer.h"


#ifndef MATRIXBENCHMARKS_XAVIERINITIALIZER_H
#define MATRIXBENCHMARKS_XAVIERINITIALIZER_H



class XavierInitializer: public BaseInitializer {
public:
    XavierInitializer(int inputUnits, int outputUnits);

    const double getValue();
protected:

    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
};


#endif //MATRIXBENCHMARKS_XAVIERINITIALIZER_H
