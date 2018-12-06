//
// Created by Martin Galajda on 26/10/2018.
//

#include "BaseInitializer.h"

#ifndef MATRIXBENCHMARKS_ZEROINITIALIZER_H
#define MATRIXBENCHMARKS_ZEROINITIALIZER_H


class ZeroInitializer: public BaseInitializer {
public:
    ZeroInitializer() {};

    const double getValue() {
        return 0.0;
    }
};


#endif //MATRIXBENCHMARKS_ZEROINITIALIZER_H
