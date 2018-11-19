//
// Created by Martin Galajda on 28/10/2018.
//

#include "gtest/gtest.h"
#include "../matrix_impl/Matrix.hpp"
#include "../data/MNISTParser.h"
#include "../neural_network/ComputationalGraph.h"
#include "../utilities/split_to_test_and_validation.tpp"
#include "../utilities/populate_placeholders.h"
#include "../neural_network/MiniBatchOptimizer.h"
#include "../neural_network/MomentumOptimizer.h"
#include "../initializers/XavierInitializer.h"
#include "Matrix.cpp"
#include <cmath>
#include <random>


double computeAccuracy(
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

    auto t_end = std::chrono::high_resolution_clock::now();
    auto secondsPassed = std::chrono::duration<double>(t_end-t_start).count();
    return secondsPassed;
}

TEST(mnist, solution1)
{
    // get training data
    MNISTParser parser("../data/mnist_train_vectors.csv", "../data/mnist_train_labels.csv");
    auto all_instances = parser.parseToMatrices();
    auto all_labels = parser.parseLabelsToOneHotEncodedVectors();

    const double VALIDATION_SET_SIZE = 0.2;
    auto indices = splitToTestAndValidationSetIndices(all_instances, VALIDATION_SET_SIZE);
    std::vector<int> validation_indices = indices["validation"];
    std::vector<int> training_indices = indices["training"];

    const int BATCH_SIZE = 35;

    const int L1_NUM_OF_NEURONS = 512;
    const int L2_NUM_OF_NEURONS = 256;
    const int OUTPUT_CLASSES = all_labels[0]->getNumOfRows();
    const int INPUT_DIMENSIONS = all_instances[0]->getNumOfRows();

    auto initializer = new XavierInitializer(INPUT_DIMENSIONS, L1_NUM_OF_NEURONS);
    auto initializer2 = new XavierInitializer(L1_NUM_OF_NEURONS, L2_NUM_OF_NEURONS);
    auto initializer4 = new XavierInitializer(L2_NUM_OF_NEURONS, OUTPUT_CLASSES);

    auto inputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(INPUT_DIMENSIONS, BATCH_SIZE));
    auto expectedOutputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(OUTPUT_CLASSES, BATCH_SIZE));

    ComputationalGraph computationalGraph;

    // First layer - hidden
    computationalGraph.addLayer(
            {
                    { "width", INPUT_DIMENSIONS },
                    { "height", L1_NUM_OF_NEURONS },
                    { "batchSize", BATCH_SIZE }
            },
            initializer,
            ActivationFunction::relu);

    // Second layer - hidden
    computationalGraph.addLayer(
            {
                    { "width", L1_NUM_OF_NEURONS },
                    { "height", L2_NUM_OF_NEURONS },
                    { "batchSize", BATCH_SIZE }
            },
            initializer2,
            ActivationFunction::relu);

    computationalGraph.addLayer(
            {
                    { "width", L2_NUM_OF_NEURONS },
                    { "height", OUTPUT_CLASSES },
                    { "batchSize", BATCH_SIZE }
            },
            initializer4,
            ActivationFunction::softmax);

    double learningRate = 10;
    MiniBatchOptimizer optimizer(computationalGraph, all_instances, all_labels, training_indices, BATCH_SIZE, learningRate);

    auto BATCHES_REQUIRED = all_instances.size() / BATCH_SIZE;
    std::cout << std::endl << "Batches required: " << BATCHES_REQUIRED << std::endl;

    std::map<double, bool > stageAlreadyLogged = {
            { 60*5.0, false },
            { 60*10.0, false },
            { 60*15.0, false },
            { 60*20.0, false }
    };


    double secondsComputingAccuracy = 0.0;
    auto t_start = std::chrono::high_resolution_clock::now();
    double secondsPassed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();

    int batch = 0;
    for (batch = 0; secondsPassed < (60*25); batch++) {
        optimizer.train();

        for (auto &stageElement: stageAlreadyLogged) {
            if (secondsPassed > stageElement.first && !stageElement.second) {
                secondsComputingAccuracy += computeAccuracy(
                        inputs,
                        all_instances,
                        expectedOutputs,
                        all_labels,
                        computationalGraph,
                        BATCH_SIZE,
                        validation_indices
                );

                std::cout << "Seconds passed: " << secondsPassed << std::endl;
                std::cout << std::endl << "Processed examples: " << batch * BATCH_SIZE << std::endl;

                stageElement.second = true;
            }
        }


        secondsPassed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count() - secondsComputingAccuracy;
    }

    std::cout << std::endl << "Processed examples: " << batch * BATCH_SIZE << std::endl;
    std::cout << std::endl << "Validation computing (s): " << secondsComputingAccuracy << std::endl;

    std::cout << "Validation ";
    computeAccuracy(
            inputs,
            all_instances,
            expectedOutputs,
            all_labels,
            computationalGraph,
            BATCH_SIZE,
            validation_indices
    );

    std::cout << "Train ";
    computeAccuracy(
            inputs,
            all_instances,
            expectedOutputs,
            all_labels,
            computationalGraph,
            BATCH_SIZE,
            training_indices
    );

}

TEST(mnist, solution2)
{
    // get training data
    MNISTParser parser("../data/mnist_train_vectors.csv", "../data/mnist_train_labels.csv");
    auto all_instances = parser.parseToMatrices();
    auto all_labels = parser.parseLabelsToOneHotEncodedVectors();

    // split training data to train and validation sets
    const double VALIDATION_SET_SIZE = 0.1;
    auto indices = splitToTestAndValidationSetIndices(all_instances, VALIDATION_SET_SIZE);
    std::vector<int> validation_indices = indices["validation"];
    std::vector<int> training_indices = indices["training"];

    const int BATCH_SIZE = 100;

    const int L1_NUM_OF_NEURONS = 32;
    const int L2_NUM_OF_NEURONS = 32;
    const int L3_NUM_OF_NEURONS = 32;
    const int L4_NUM_OF_NEURONS = 32;
    const int L5_NUM_OF_NEURONS = 16;
    const int L6_NUM_OF_NEURONS = 16;

    const int OUTPUT_CLASSES = all_labels[0]->getNumOfRows();
    const int INPUT_DIMENSIONS = all_instances[0]->getNumOfRows();

    auto initializerOutput = new XavierInitializer(L4_NUM_OF_NEURONS, OUTPUT_CLASSES);

    auto inputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(INPUT_DIMENSIONS, BATCH_SIZE));
    auto expectedOutputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(OUTPUT_CLASSES, BATCH_SIZE));

    std::map<int, std::map<std::string, int>> denseLayerConfigMap = {
            { 1, {{ "width", INPUT_DIMENSIONS  }, { "height", L1_NUM_OF_NEURONS }}},
            { 2, {{ "width", L1_NUM_OF_NEURONS }, { "height", L2_NUM_OF_NEURONS }}},
            { 3, {{ "width", L2_NUM_OF_NEURONS }, { "height", L3_NUM_OF_NEURONS }}},
            { 4, {{ "width", L3_NUM_OF_NEURONS }, { "height", L4_NUM_OF_NEURONS }}},
            { 5, {{ "width", L4_NUM_OF_NEURONS }, { "height", L5_NUM_OF_NEURONS }}},
            { 6, {{ "width", L5_NUM_OF_NEURONS }, { "height", L6_NUM_OF_NEURONS }}},
    };

    ComputationalGraph computationalGraph;


    for (auto i = 1; i <= denseLayerConfigMap.size(); i++) {
        auto config = denseLayerConfigMap[i];
        auto initializer = new XavierInitializer(config["width"],config["height"]);

        computationalGraph.addLayer(
                {
                        { "width", config["width"] },
                        { "height", config["height"] },
                        { "batchSize", BATCH_SIZE }
                },
                initializer,
                ActivationFunction::relu);

        free(initializer);
    }

    // Output layer - output
    computationalGraph.addLayer(
            {
                    { "width", L6_NUM_OF_NEURONS },
                    { "height", OUTPUT_CLASSES },
                    { "batchSize", BATCH_SIZE }
            },
            initializerOutput,
            ActivationFunction::softmax);


    auto BATCHES_REQUIRED = all_instances.size() / BATCH_SIZE;

    std::cout
            << std::endl
            << "Batches required: "
            << BATCHES_REQUIRED
            << std::endl;

    double learningRate = 0.1;

    MomentumOptimizer optimizer(
            computationalGraph,
            all_instances,
            all_labels,
            training_indices,
            BATCH_SIZE,
            learningRate
    );

    std::map<double, bool > stageAlreadyLogged = {
            { 60*5.0, false },
            { 60*10.0, false },
            { 60*15.0, false },
            { 60*20.0, false }
    };


    double secondsComputingAccuracy = 0.0;
    auto t_start = std::chrono::high_resolution_clock::now();
    double secondsPassed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();

    auto batch = 0;
    for (; secondsPassed < (60*25); batch++) {
        optimizer.train();

        for (auto &stageElement: stageAlreadyLogged) {
            if (secondsPassed > stageElement.first && !stageElement.second) {
                secondsComputingAccuracy += computeAccuracy(
                        inputs,
                        all_instances,
                        expectedOutputs,
                        all_labels,
                        computationalGraph,
                        BATCH_SIZE,
                        validation_indices
                );

                std::cout << "Seconds passed: " << secondsPassed << std::endl;

                stageElement.second = true;
            }
        }


        secondsPassed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count() - secondsComputingAccuracy;
    }

    std::cout << std::endl << "Processed examples: " << batch * BATCH_SIZE << std::endl;
    std::cout << std::endl << "Validation computing (s): " << secondsComputingAccuracy << std::endl;

    std::cout << "Validation ";
    computeAccuracy(
            inputs,
            all_instances,
            expectedOutputs,
            all_labels,
            computationalGraph,
            BATCH_SIZE,
            validation_indices
    );

    std::cout << "Train ";
    computeAccuracy(
            inputs,
            all_instances,
            expectedOutputs,
            all_labels,
            computationalGraph,
            BATCH_SIZE,
            training_indices
    );


    free(initializerOutput);

}