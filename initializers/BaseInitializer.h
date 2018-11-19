//
// Created by Martin Galajda on 26/10/2018.
//
#define _GLIBCXX_USE_CXX11_ABI 0

#ifndef MATRIXBENCHMARKS_BASEINITIALIZER_H
#define MATRIXBENCHMARKS_BASEINITIALIZER_H


class BaseInitializer {
public:

    virtual const double getValue() = 0;

protected:
};


#endif //MATRIXBENCHMARKS_BASEINITIALIZER_H
