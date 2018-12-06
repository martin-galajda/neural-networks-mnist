#include <iostream>
#include "matrix_impl/Matrix.hpp"
#include "data/MNISTParser.h"
#include "neural_network/ComputationalGraph.h"
#include "utilities/split_to_test_and_validation.tpp"
#include "utilities/populate_placeholders.h"
#include "neural_network/MiniBatchOptimizer.h"
#include "neural_network/MomentumOptimizer.h"
#include "initializers/XavierInitializer.h"
#include <cmath>
#include <random>
#include <chrono>
#include <memory>
#include <algorithm>    // std::max
#include "./enums.h"
#include "./utilities/config_params_utils.h"

#define _GLIBCXX_USE_CXX11_ABI 0

//
std::map<std::string, double> computeAccuracy(
        std::shared_ptr<Matrix<double>> &inputs,
        std::vector<std::shared_ptr<Matrix<double>>> &instances,
        std::shared_ptr<Matrix<double>> &expectedOutputs,
        std::vector<std::shared_ptr<Matrix<double>>> &labels,
        ComputationalGraph &graph,
        const int &BATCH_SIZE,
        std::vector<int> &indicesPool
) {
    auto t_start = std::chrono::high_resolution_clock::now();
    const int NUM_OF_INSTANCES = indicesPool.size();

    std::vector<int> instanceIndexes;
    instanceIndexes.reserve(BATCH_SIZE);

    auto matchedPrediction = 0.0;

    for (auto i = 0; i < NUM_OF_INSTANCES - BATCH_SIZE; i += BATCH_SIZE) {
        instanceIndexes.clear();
        instanceIndexes.reserve(BATCH_SIZE);
        for (auto j = i; j < i + BATCH_SIZE && j < NUM_OF_INSTANCES; j++) {
            instanceIndexes.push_back(indicesPool[j]);
        }

        populatePlaceholders(inputs, instances, instanceIndexes);
        populatePlaceholders(expectedOutputs, labels, instanceIndexes);

        auto s = graph.forwardPass(inputs);

        auto predictedValues = s->argMaxByRow();
        auto expectedValues = expectedOutputs->argMaxByRow();

        for (auto row = 0; row < predictedValues->getNumOfRows(); row++) {
            if ((*predictedValues)[row][0] == (*expectedValues)[row][0]) {
                matchedPrediction++;
            }
        }
    }

    std::cout << "Accuracy: "
              << ((matchedPrediction * 1.0) / NUM_OF_INSTANCES)
              << std::endl;
    std::cout << "Matched predictions: "
              << matchedPrediction * 1.0;
    std::cout << ". Out of: "
              << NUM_OF_INSTANCES
              << std::endl;

    auto t_end = std::chrono::high_resolution_clock::now();
    auto secondsPassed = std::chrono::duration<double>(t_end-t_start).count();
    return {
        { "secondsPassed", secondsPassed },
        { "accuracy", ((matchedPrediction * 1.0) / NUM_OF_INSTANCES) }
    };
}


std::map<std::string, double> performTraining(
        std::map<int, int> &generalConfig,
        std::map<std::string, double> &generalCoeffConfig,
        std::map<int, std::map<std::string, int>> &denseLayerConfigMap,
        std::vector<std::shared_ptr<Matrix<double>>> &all_instances,
        std::vector<std::shared_ptr<Matrix<double>>> &all_labels,
        std::vector<int> &training_indices,
        std::vector<int> &validation_indices
) {
    double secondsComputingAccuracy = 0.0;
    auto t_start = std::chrono::high_resolution_clock::now();
    double secondsPassed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();


    double learningRateStart;
    double learningRateEnd;
    double decayAfterEpochs = 1.0;

    if (generalConfig[generalConfigEnums::ConfigType::learningRateStrategy] == generalConfigEnums::LearningRateStrategy::flat) {
        learningRateStart = generalCoeffConfig["learningRate"];
        learningRateEnd = generalCoeffConfig["learningRate"];
    } else {
        learningRateStart = generalCoeffConfig["learningRateStart"];
        learningRateEnd = generalCoeffConfig["learningRateEnd"];
        decayAfterEpochs = generalCoeffConfig["learningRateDecayAfterEpochs"];
    }

    auto BATCH_SIZE = generalCoeffConfig["batchSize"];

    auto BATCHES_REQUIRED = training_indices.size() / BATCH_SIZE;

    const double learningRateDecayWeight = (learningRateEnd - learningRateStart) / (decayAfterEpochs * BATCHES_REQUIRED);
    auto currLearningRate = learningRateStart;
    int batch = 0;

    const double l2rate = generalCoeffConfig["l2reg"];

    BaseInitializer *initializer;

    ComputationalGraph *computationalGraphPtr = new ComputationalGraph();
    ComputationalGraph &computationalGraph = *computationalGraphPtr;

    const auto INPUT_DIMENSIONS = denseLayerConfigMap[1]["width"];
    const auto OUTPUT_CLASSES = denseLayerConfigMap[denseLayerConfigMap.size()]["height"];

    auto inputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(INPUT_DIMENSIONS, BATCH_SIZE));
    auto expectedOutputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(OUTPUT_CLASSES, BATCH_SIZE));

    for (auto i = 1; i <= denseLayerConfigMap.size(); i++) {
        auto config = denseLayerConfigMap[i];

        if (generalConfig[generalConfigEnums::ConfigType::initializer] == generalConfigEnums::xavier) {
            initializer = new XavierInitializer(config["width"],config["height"]);
        } else {
            initializer = new NormalInitializer();
        }

        auto activationFunction = ActivationFunction ::relu;

        if (config.find("activation") != config.end()) {
            activationFunction = (ActivationFunction) config.find("activation")->second;
            std::cout<<activationFunction<<std::endl;
        }

        double l2reg = l2rate;

        if (config.find("l2reg") != config.end()) {
            l2reg = (double) config.find("l2reg")->second;
        }

        computationalGraph.addLayer(
                {
                        { "width", config["width"] },
                        { "height", config["height"] },
                        { "batchSize", generalCoeffConfig["batchSize"] }
                },
                initializer,
                activationFunction,
                l2reg
        );

        free(initializer);
    }

    BaseOptimizer *optimizerPtr;

    if (generalConfig[generalConfigEnums::ConfigType::optimizer] == generalConfigEnums::momentum) {
        optimizerPtr = new MomentumOptimizer(computationalGraph, all_instances, all_labels, training_indices, BATCH_SIZE, currLearningRate);
    } else if (generalConfig[generalConfigEnums::ConfigType::optimizer] == generalConfigEnums::minibatch) {
        optimizerPtr = new MiniBatchOptimizer(computationalGraph, all_instances, all_labels, training_indices, BATCH_SIZE, currLearningRate);;
    }
    auto &optimizer = *optimizerPtr;

    for (batch = 0; secondsPassed < (60 * generalCoeffConfig["maxTrainTimeMinutes"]); batch++) {
        optimizerPtr->train();

        if (currLearningRate > learningRateEnd) {
            currLearningRate -= learningRateDecayWeight;
        }
        currLearningRate = std::max(currLearningRate, learningRateEnd);
        optimizerPtr->setLearningRate(currLearningRate);

        secondsPassed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count() - secondsComputingAccuracy;
    }

    std::cout << std::endl << "Processed examples: " << batch * BATCH_SIZE << std::endl;
    std::cout << std::endl << "Validation computing (s): " << secondsComputingAccuracy << std::endl;

    std::cout << "Validation ";
    auto validationAccuracy = computeAccuracy(
            inputs,
            all_instances,
            expectedOutputs,
            all_labels,
            computationalGraph,
            BATCH_SIZE,
            validation_indices
    )["accuracy"];

    std::cout << "Train ";
    auto trainAccuracy = computeAccuracy(
            inputs,
            all_instances,
            expectedOutputs,
            all_labels,
            computationalGraph,
            BATCH_SIZE,
            training_indices
    )["accuracy"];

    std::map<std::string, double> results = {
            { "validationAccuracy", validationAccuracy },
            { "trainingAccuracy", trainAccuracy },
    };
    std::cout << makeConfigParamsString(generalConfig, generalCoeffConfig, computationalGraph, results) << std::endl;

    return {
            { "secondsComputingAccuracy", secondsComputingAccuracy },
            { "validationAccuracy", validationAccuracy },
            { "trainAccuracy", trainAccuracy },
    };
}

int main(int argc, const char** argv) {
    // get training data
    MNISTParser parser("../data/mnist_train_vectors.csv", "../data/mnist_train_labels.csv");
    auto all_instances = parser.parseToMatrices();
    auto all_labels = parser.parseLabelsToOneHotEncodedVectors();

    const double VALIDATION_SET_SIZE = 0.1;
    auto indices = splitToTestAndValidationSetIndices(all_instances, VALIDATION_SET_SIZE);
    std::vector<int> validation_indices = indices["validation"];
    std::vector<int> training_indices = indices["training"];

    const int BATCH_SIZE = 27;

    const int OUTPUT_CLASSES = all_labels[0]->getNumOfRows();
    const int INPUT_DIMENSIONS = all_instances[0]->getNumOfRows();

    auto inputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(INPUT_DIMENSIONS, BATCH_SIZE));
    auto expectedOutputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(OUTPUT_CLASSES, BATCH_SIZE));

    auto l2reg = 0.000001;
    std::map<int, std::map<std::string, int>> denseLayerConfigMap = {
            { 1, {{ "width", INPUT_DIMENSIONS  }, { "height", 100 }, {"l2reg", 0 }}},
            { 2, {{ "width", 100 }, {"height", 121 }}},
            { 3, {{ "width", 121 }, {"height", 26 }}},
            { 4, {{ "width", 26 }, {"height", 163 }}},
            { 5, {{ "width", 163 }, {"height", 218 }}},
            { 6, {{ "width", 218 }, {"height", 110 }}},
            { 7, {{ "width", 110 }, { "height", OUTPUT_CLASSES }, {"activation", ActivationFunction ::softmax}, {"l2reg", 0 }}}
    };

    const double l2rate = 0.000001;
    double learningRate = 0.00749615;
    const double learningRateEnd = 0.00452354;

    std::map<int, int> generalConfig = {
            { generalConfigEnums::ConfigType ::learningRateStrategy, generalConfigEnums::LearningRateStrategy ::constantDecay },
            { generalConfigEnums::ConfigType ::initializer, generalConfigEnums::Initializer ::xavier},
            { generalConfigEnums::ConfigType ::optimizer, generalConfigEnums::Optimizer ::momentum}
    };

    std::map<std::string, double> generalCoeffConfig = {
            { "maxTrainTimeMinutes", 28 },
            { "batchSize", BATCH_SIZE },
            { "l2reg" , l2rate },
            { "learningRateStart", learningRate },
            { "learningRateEnd", learningRateEnd },
            { "learningRate", learningRate },
            { "learningRateDecayAfterEpochs", 10 },
    };

    performTraining(
            generalConfig,
            generalCoeffConfig,
            denseLayerConfigMap,
            all_instances,
            all_labels,
            training_indices,
            validation_indices
    );


    return 0;
}
