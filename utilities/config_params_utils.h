//
// Created by Martin Galajda on 01/11/2018.
//
#define _GLIBCXX_USE_CXX11_ABI 0


#include <sstream>
#include <map>
#include "../neural_network/ComputationalGraph.h"

#ifndef UTILITIES_CONFIG_PARAMS_UTILS_H
#define UTILITIES_CONFIG_PARAMS_UTILS_H

std::string makeConfigParamsString(
        std::map<int, int> &generalConfig,
        std::map<std::string, double> &generalCoeffConfig,
        ComputationalGraph &graph,
        std::map<std::string, double> &results
);

#endif //UTILITIES_CONFIG_PARAMS_UTILS_H
