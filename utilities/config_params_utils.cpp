//
// Created by Martin Galajda on 01/11/2018.
//
#define _GLIBCXX_USE_CXX11_ABI 0

#include "config_params_utils.h"
#include "../enums.h"


inline const char* optimizerEnumToString(int optimizer) {
    switch (optimizer)
    {
        case generalConfigEnums::Optimizer::momentum:   return "momentum";
        case generalConfigEnums::Optimizer::minibatch:   return "minibatch";
        case generalConfigEnums::Optimizer::adam:   return "adam";
        default:      return "[Unknown optimizer]";
    }
}

inline const char* initializerEnumToString(int initializer) {
    switch (initializer)
    {
        case generalConfigEnums::Initializer ::xavier:  return "xavier";
        default:      return "[Unknown initializer]";
    }
}
inline const char* learningStrategyEnumToString(int learningRateStrategy) {
    switch (learningRateStrategy)
    {
        case generalConfigEnums::LearningRateStrategy ::constantDecay:   return "constantDecay";
        case generalConfigEnums::LearningRateStrategy ::flat:   return "flat";
        default:      return "[Unknown learningStrategy]";
    }
}


std::string makeConfigParamsString(
    std::map<int, int> &generalConfig,
    std::map<std::string, double> &generalCoeffConfig,
    ComputationalGraph &graph,
    std::map<std::string, double> &results
) {
    std::stringstream ss;

    auto learningRateStrategy = generalConfig[generalConfigEnums::ConfigType::learningRateStrategy];

    ss << learningStrategyEnumToString(learningRateStrategy) << ",";

    if (learningRateStrategy == generalConfigEnums::LearningRateStrategy ::constantDecay) {
        auto learningRateStart = generalCoeffConfig["learningRateStart"];
        auto learningRateEnd = generalCoeffConfig["learningRateEnd"];
        ss << learningRateStart << "," <<  learningRateEnd << ",";
    } else if (learningRateStrategy == generalConfigEnums::LearningRateStrategy ::flat) {
        auto learningRateStart = generalCoeffConfig["learningRate"];
        auto learningRateEnd = generalCoeffConfig["learningRate"];
        ss << learningRateStart << "," <<  learningRateEnd << ",";
    }


    ss << generalCoeffConfig["batchSize"] << ","
        << generalCoeffConfig["l2reg"] << ","
        << initializerEnumToString(generalConfig[generalConfigEnums::ConfigType::initializer]) << ","
        << generalCoeffConfig["maxTrainTimeMinutes"] << ","
        << optimizerEnumToString(generalConfig[generalConfigEnums::ConfigType::optimizer]) << ",";

    auto layers = graph.getLayers();
    for (auto &layer : layers) {
        ss << "[" << layer->getWidth() << ";" << layer->getHeight() << "] ";
    }
    ss << ",";

    ss << results["trainingAccuracy"] << "," << results["validationAccuracy"]
        << results["testAccuracy"];

    return ss.str();
}
