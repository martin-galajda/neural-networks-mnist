#include <iostream>
#include "matrix_impl/Matrix.hpp"
#include "data/MNISTParser.h"
#include "neural_network/ComputationalGraph.h"
#include "utilities/split_to_test_and_validation.tpp"
#include "utilities/populate_placeholders.h"
#include "neural_network/MiniBatchOptimizer.h"
#include "neural_network/MomentumOptimizer.h"
#include "neural_network/AdamOptimizer.h"
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


std::shared_ptr<ComputationalGraph> performTraining(
        std::map<int, int> &generalConfig,
        std::map<std::string, double> &generalCoeffConfig,
        std::map<int, std::map<std::string, int>> &denseLayerConfigMap,
        std::vector<std::shared_ptr<Matrix<double>>> &all_instances,
        std::vector<std::shared_ptr<Matrix<double>>> &all_labels,
        std::vector<int> &training_indices,
        std::vector<int> &validation_indices,

        std::vector<std::shared_ptr<Matrix<double>>> &all_test_instances,
        std::vector<std::shared_ptr<Matrix<double>>> &all_test_labels,

        std::vector<int> &test_indices
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

    const double learningRateDecayWeight = (learningRateStart - learningRateEnd) / (decayAfterEpochs * BATCHES_REQUIRED);
    auto currLearningRate = learningRateStart;
    int batch = 0;

    const double l2rate = generalCoeffConfig["l2reg"];

    BaseInitializer *initializer;

    auto computationalGraphPtr = std::shared_ptr<ComputationalGraph>(new ComputationalGraph());
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
    } else if  (generalConfig[generalConfigEnums::ConfigType::optimizer] == generalConfigEnums::adam) {
        optimizerPtr = new AdamOptimizer(computationalGraph, all_instances, all_labels, training_indices, BATCH_SIZE, currLearningRate);;
    }
    auto &optimizer = *optimizerPtr;

    std::map<double, bool > stageAlreadyLogged = {
            { 60*0.5, false },
            { 60*1.0, false },
            { 60*1.5, false },
            { 60*3.0, false },
            { 60*5.0, false },
            { 60*7.5, false },
            { 60*10.0, false },
            { 60*12.5, false },
            { 60*15.0, false },
            { 60*17.5, false },
            { 60*20.0, false },
            { 60*22.5, false },
            { 60*30.5, false },
            { 60*35.5, false },
            { 60*40.5, false },
            { 60*45.5, false },
            { 60*50.5, false },
    };

    for (batch = 0; secondsPassed < (60 * generalCoeffConfig["maxTrainTimeMinutes"]); batch++) {
        optimizerPtr->train();

        if (currLearningRate > learningRateEnd) {
            currLearningRate -= learningRateDecayWeight;
        }
        currLearningRate = std::max(currLearningRate, learningRateEnd);
        optimizerPtr->setLearningRate(currLearningRate);

        for (auto &stageElement: stageAlreadyLogged) {
            if (secondsPassed > stageElement.first && !stageElement.second) {
//                std::cout << std::endl;
//
//                secondsComputingAccuracy += computeAccuracy(
//                        inputs,
//                        all_instances,
//                        expectedOutputs,
//                        all_labels,
//                        computationalGraph,
//                        BATCH_SIZE,
//                        validation_indices
//                )["secondsPassed"];
//
//                std::cout << "Seconds passed: " << secondsPassed << std::endl;
//                std::cout << "Processed examples: " << batch * BATCH_SIZE << std::endl;

                stageElement.second = true;
            }
        }


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


    std::cout << "Test ";

    auto testAccuracy = computeAccuracy(
            inputs,
            all_test_instances,
            expectedOutputs,
            all_test_labels,
            computationalGraph,
            BATCH_SIZE,
            test_indices
    )["accuracy"];


    std::map<std::string, double> results = {
            { "validationAccuracy", validationAccuracy },
            { "trainingAccuracy", trainAccuracy },
            { "testAccuracy", testAccuracy },
    };
    std::cout << makeConfigParamsString(generalConfig, generalCoeffConfig, computationalGraph, results) << std::endl;

//    return {
//            { "secondsComputingAccuracy", secondsComputingAccuracy },
//            { "validationAccuracy", validationAccuracy },
//            { "trainAccuracy", trainAccuracy },
//            { "testAccuracy", testAccuracy },
//    };
    return computationalGraphPtr;
}


std::map<std::string, double> makeCoeffConfig(
        std::map<int, int> &generalConfig,
        double maxTrainTimeInMinutes,
        double maxLearningRate,
        double minLearningRate,
        double maxL2,
        double minL2,
        int maxBatchSize,
        int minBatchSize
) {
    std::map<std::string, double> generalCoeffConfig;

    std::mt19937 rng(randomDevice());    // random-number engine used (Mersenne-Twister in this case)

    std::uniform_real_distribution<double> pickRandomLearningRate(minLearningRate, maxLearningRate); // guaranteed unbiased
    std::uniform_real_distribution<double> pickRandomL2Coeff(minL2, maxL2); // guaranteed unbiased
    std::uniform_int_distribution<int> pickRandomBatchSize(minBatchSize, maxBatchSize); // guaranteed unbiased

    if (generalConfig[generalConfigEnums::ConfigType::learningRateStrategy] != generalConfigEnums::constantDecay) {
        std::uniform_real_distribution<double> pickRandomLearningRate(minLearningRate, maxLearningRate); // guaranteed unbiased
        generalCoeffConfig["learningRate"] = pickRandomLearningRate(randomDevice);
        generalCoeffConfig["learningRateStart"] = generalCoeffConfig["learningRate"];
        generalCoeffConfig["learningRateEnd"] = generalCoeffConfig["learningRate"];

    } else {
        std::uniform_real_distribution<double> pickRandomLearningRate(minLearningRate, maxLearningRate); // guaranteed unbiased
        auto learningRateOne = pickRandomLearningRate(randomDevice);
        auto learningRateTwo = pickRandomLearningRate(randomDevice);
        generalCoeffConfig["learningRateStart"] = std::max(learningRateOne, learningRateTwo);
        generalCoeffConfig["learningRateEnd"] = std::min(learningRateOne, learningRateTwo);
        generalCoeffConfig["learningRateDecayAfterEpochs"] = 4;
    }

    generalCoeffConfig["maxTrainTimeMinutes"] = maxTrainTimeInMinutes;
    generalCoeffConfig["batchSize"] = pickRandomBatchSize(randomDevice);
    generalCoeffConfig["l2reg"] = pickRandomL2Coeff(randomDevice);

    std::cout << "Learning rate: " << generalCoeffConfig["learningRateStart"] <<  "-" << generalCoeffConfig["learningRateEnd"] << std::endl;

    return generalCoeffConfig;
}

std::map<int, std::map<std::string, int>> makeDenseLayerConfig(
    int MIN_NUMBER_OF_LAYERS,
    int MAX_NUMBER_OF_LAYERS,
    int MIN_NEURONS_LAYER,
    int MAX_NEURONS_LAYER,
    std::map<int, int> &generalConfig,
    std::map<std::string, double> &coeffConfig
) {
    std::uniform_int_distribution<int> pickRandomNumberOfLayers(MIN_NUMBER_OF_LAYERS, MAX_NUMBER_OF_LAYERS); // guaranteed unbiased
    std::uniform_int_distribution<int> pickRandomNumberOfNeurons(MIN_NEURONS_LAYER, MAX_NEURONS_LAYER); // guaranteed unbiased

    auto numberOfLayers = pickRandomNumberOfLayers(randomDevice);
    std::map<int, std::map<std::string, int>> denseLayerConfigMap;

    auto lastLayerHeight = generalConfig[generalConfigEnums::ConfigType::inputDimensions];
    for (auto i = 1; i < numberOfLayers; i++) {
        auto layerWidth = lastLayerHeight;
        auto layerHeight = pickRandomNumberOfNeurons(randomDevice);

        denseLayerConfigMap[i] = { {"width", layerWidth}, {"height", layerHeight}, {"l2reg", coeffConfig["l2reg"] }};
        lastLayerHeight = layerHeight;
    }

    denseLayerConfigMap[numberOfLayers] = {
            {"width", lastLayerHeight},
            {"height", generalConfig[generalConfigEnums::ConfigType::outputDimensions]},
            {"activation", ActivationFunction ::softmax},
            {"l2reg", 0 }
    };

    return denseLayerConfigMap;
}


std::map<int, int> makeGeneralConfig(
        std::vector<std::shared_ptr<Matrix<double>>> &all_instances,
        std::vector<std::shared_ptr<Matrix<double>>> &all_labels
) {

    std::map<int, int> generalConfig;
    std::mt19937 rng(randomDevice());    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> pickRandomInt(0, 1); // guaranteed unbiased

    int learningRateStrategy = pickRandomInt(rng);

    int initializer = generalConfigEnums::xavier;

    std::uniform_int_distribution<int> pickRandomOptimizer(0, 2); // guaranteed unbiased
    int optimizer = pickRandomOptimizer(rng);

    generalConfig[generalConfigEnums::ConfigType::learningRateStrategy] = learningRateStrategy;
    generalConfig[generalConfigEnums::ConfigType::optimizer] = optimizer;
    generalConfig[generalConfigEnums::ConfigType::initializer] = initializer;

    const int OUTPUT_CLASSES = all_labels[0]->getNumOfRows();
    const int INPUT_DIMENSIONS = all_instances[0]->getNumOfRows();

    generalConfig[generalConfigEnums::ConfigType::outputDimensions] = OUTPUT_CLASSES;
    generalConfig[generalConfigEnums::ConfigType::inputDimensions] = INPUT_DIMENSIONS;

    return generalConfig;
}

std::vector<
    std::tuple<std::map<int, int>, std::map<std::string, double>, std::map<int, std::map<std::string, int>>>
> generateConfigs(
    int numberOfConfigs,
    int MIN_NUMBER_OF_LAYERS,
    int MAX_NUMBER_OF_LAYERS,
    int MIN_NEURONS_LAYER,
    int MAX_NEURONS_LAYER,

    double maxTrainTimeInMinutes,
    double maxLearningRate,
    double minLearningRate,
    double maxL2,
    double minL2,
    int maxBatchSize,
    int minBatchSize,

    std::vector<std::shared_ptr<Matrix<double>>> &all_instances,
    std::vector<std::shared_ptr<Matrix<double>>> &all_labels
) {

    std::vector<
            std::tuple<std::map<int, int>, std::map<std::string, double>, std::map<int, std::map<std::string, int>>>
    > configs;

    for (auto i = 0; i < numberOfConfigs; i++) {
        auto generalConfig = makeGeneralConfig(
                all_instances,
                all_labels
        );

        auto coeffConfig = makeCoeffConfig(
                generalConfig,
                maxTrainTimeInMinutes,
                maxLearningRate,
                minLearningRate,
                maxL2,
                minL2,
                maxBatchSize,
                minBatchSize
        );

        auto denseLayerConfig = makeDenseLayerConfig(
                MIN_NUMBER_OF_LAYERS,
                MAX_NUMBER_OF_LAYERS,
                MIN_NEURONS_LAYER,
                MAX_NEURONS_LAYER,
                generalConfig,
                coeffConfig
        );

        auto newConfig = std::make_tuple(generalConfig, coeffConfig, denseLayerConfig);
        configs.push_back(newConfig);

        // compilers have some problem with this on aisa...
        // configs.push_back({ generalConfig, coeffConfig, denseLayerConfig });
    }

    return configs;
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

    MNISTParser testDataParser("../data/mnist_test_vectors.csv", "../data/mnist_test_labels.csv");
    auto all_test_instances = testDataParser.parseToMatrices();
    auto all_test_labels = testDataParser.parseLabelsToOneHotEncodedVectors();
    std::vector<int> test_indices(all_test_instances.size());
    std::iota(test_indices.begin(), test_indices.end(), 0);


    int numberOfConfigs = 20;
    int MIN_NUMBER_OF_LAYERS = 2;
    int MAX_NUMBER_OF_LAYERS = 3;
    int MIN_NEURONS_LAYER = 10;
    int MAX_NEURONS_LAYER = 100;

    double maxTrainTimeInMinutes = 1;
    double maxLearningRate = 0.05;
    double minLearningRate = 0.0001;
    double maxL2 = 0.0001;
    double minL2 = 0;
    int maxBatchSize = 100;
    int minBatchSize = 10;


    auto configs = generateConfigs(
            numberOfConfigs,
            MIN_NUMBER_OF_LAYERS,
            MAX_NUMBER_OF_LAYERS,
            MIN_NEURONS_LAYER,
            MAX_NEURONS_LAYER,

            maxTrainTimeInMinutes,
            maxLearningRate,
            minLearningRate,
            maxL2,
            minL2,
            maxBatchSize,
            minBatchSize,

            all_instances,
            all_labels
    );

    for (auto config: configs) {
        performTraining(
                std::get<0>(config),
                std::get<1>(config),
                std::get<2>(config),
                all_instances,
                all_labels,
                training_indices,
                validation_indices,

                all_test_instances,
                all_test_labels,
                test_indices
        );
    }


    return 0;
}