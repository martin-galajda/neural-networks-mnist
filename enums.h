//
// Created by Martin Galajda on 01/11/2018.
//

#include <string>

#ifndef MATRIXBENCHMARKS_ENUMS_H
#define MATRIXBENCHMARKS_ENUMS_H

namespace generalConfigEnums {
    enum Optimizer { momentum = 0, minibatch = 1, adam };

    enum LearningRateStrategy { constantDecay = 0, flat = 1 };

    enum Initializer { xavier = 0 };

    enum ConfigType { learningRateStrategy = 0, optimizer, initializer, inputDimensions, outputDimensions };

}



#endif //MATRIXBENCHMARKS_ENUMS_H
