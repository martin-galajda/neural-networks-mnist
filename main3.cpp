//
// Created by Martin Galajda on 01/12/2018.
//

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
#include "initializers/ZeroInitializer.h"
#include <cmath>
#include <random>
#include <chrono>
#include <memory>
#include <algorithm>    // std::max
#include "./enums.h"
#include "./utilities/config_params_utils.h"
#include "layers/ConvLayer.h"
#include "layers/FlattenLayer.h"
#include "layers/MaxPool2DLayer.h"

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

    populatePlaceholders2D(inputs, instances, instanceIndexes);
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


  auto BATCH_SIZE = 32;

//  auto BATCHES_REQUIRED = training_indices.size() / BATCH_SIZE;

  int batch = 0;

//  BaseInitializer *initializer;

  auto INPUT_SIZE = 28 * 28;
  auto OUTPUT_SIZE = 10;

  auto computationalGraphPtr = std::shared_ptr<ComputationalGraph>(new ComputationalGraph(28, 28, OUTPUT_SIZE));
  ComputationalGraph &computationalGraph = *computationalGraphPtr;

  auto inputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(28, 28, 1, BATCH_SIZE));
  auto expectedOutputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(10, 1, 1, BATCH_SIZE));

//  auto zeroInitializer = new ZeroInitializer();


//  Conv2D(filters=2, kernel_size=3, activation='relu', input_shape=(28,28, 1), strides=1),
//    Conv2D(filters=8, kernel_size=3, activation='relu', strides=2),
//    Conv2D(filters=64, kernel_size=3, activation='relu', strides=2),
//    Conv2D(filters=64, kernel_size=3, activation='relu', strides=2),
//    Conv2D(filters=64, kernel_size=2, activation='relu', strides=1),
//
////#         Conv2D(filters=3, kernel_size=3, activation='relu', strides=3),
//
//    Flatten(),
//    Dense(units=256, activation='relu'),
//
//    Dense(units=10, activation='softmax')


  auto STRIDE_CONV_1 = 1;
  auto STRIDE_CONV_2 = 2;
  auto STRIDE_CONV_3 = 2;
  auto STRIDE_CONV_4 = 2;
  auto STRIDE_CONV_5 = 1;
//  auto STRIDE_CONV_4 = 3;

  auto WIDTH_CONV_1 = 3;
  auto WIDTH_CONV_2 = 3;
  auto WIDTH_CONV_3 = 3;
  auto WIDTH_CONV_4 = 3;
  auto WIDTH_CONV_5 = 2;
//  auto WIDTH_CONV_4 = 5;

  auto FILTERS_CONV_1 = 2;
  auto FILTERS_CONV_2 = 4;
  auto FILTERS_CONV_3 = 32;
  auto FILTERS_CONV_4 = 32;
  auto FILTERS_CONV_5 = 32;
//  auto FILTERS_CONV_4 = 10;

  auto SIZE_CONV_1 = (int) (((28 - WIDTH_CONV_1) / STRIDE_CONV_1) + 1);
  auto SIZE_CONV_2 = (int) ((SIZE_CONV_1 - WIDTH_CONV_2) / STRIDE_CONV_2) + 1;
  auto SIZE_CONV_3 = (int) ((SIZE_CONV_2 - WIDTH_CONV_3) / STRIDE_CONV_3) + 1;
  auto SIZE_CONV_4 = (int) ((SIZE_CONV_3 - WIDTH_CONV_4) / STRIDE_CONV_4) + 1;
  auto SIZE_CONV_5 = (int) ((SIZE_CONV_4 - WIDTH_CONV_5) / STRIDE_CONV_5) + 1;
//  auto SIZE_CONV_4 = (int) ((SIZE_CONV_3 - WIDTH_CONV_4) / STRIDE_CONV_4) + 1;



  auto MAX_POOL_1_WIDTH = 4;
  auto MAX_POOL_1_HEIGHT = 4;
  auto MAX_POOL_1_STRIDE = 4;
//  auto SIZE_MAX_POOL_1 = (int) ((SIZE_CONV_3 - MAX_POOL_1_WIDTH) / MAX_POOL_1_STRIDE) + 1;
  auto SIZE_MAX_POOL_1 = (int) ((SIZE_CONV_2 - MAX_POOL_1_WIDTH) / MAX_POOL_1_STRIDE) + 1;


  auto xavierConv1 = new XavierInitializer(28 * 28, SIZE_CONV_1 * SIZE_CONV_1 * FILTERS_CONV_1);
  auto xavierConv2 = new XavierInitializer(SIZE_CONV_1 * SIZE_CONV_1 * FILTERS_CONV_1, SIZE_CONV_2 * SIZE_CONV_2 * FILTERS_CONV_2);
  auto xavierConv3 = new XavierInitializer(SIZE_CONV_2 * SIZE_CONV_2 * FILTERS_CONV_2, SIZE_CONV_3 * SIZE_CONV_3 * FILTERS_CONV_3);
  auto xavierConv4 = new XavierInitializer(SIZE_CONV_3 * SIZE_CONV_3 * FILTERS_CONV_3, SIZE_CONV_4 * SIZE_CONV_4 * FILTERS_CONV_4);
  auto xavierConv5 = new XavierInitializer(SIZE_CONV_4 * SIZE_CONV_4 * FILTERS_CONV_4, SIZE_CONV_5 * SIZE_CONV_5 * FILTERS_CONV_5);
//  auto xavierConv1 = new NormalInitializer();
//  auto xavierConv2 = new NormalInitializer();


  auto conv1 = new ConvLayer(WIDTH_CONV_1, WIDTH_CONV_1, BATCH_SIZE, 1, FILTERS_CONV_1, xavierConv1, ActivationFunction::relu, STRIDE_CONV_1, "ConvLayer1");
  auto conv2 = new ConvLayer(WIDTH_CONV_2, WIDTH_CONV_2, BATCH_SIZE, FILTERS_CONV_1, FILTERS_CONV_2, xavierConv2, ActivationFunction::relu, STRIDE_CONV_2, "ConvLayer2");
  auto conv3 = new ConvLayer(WIDTH_CONV_3, WIDTH_CONV_3, BATCH_SIZE, FILTERS_CONV_2, FILTERS_CONV_3, xavierConv3, ActivationFunction::relu, STRIDE_CONV_3, "ConvLayer3");
  auto conv4 = new ConvLayer(WIDTH_CONV_4, WIDTH_CONV_4, BATCH_SIZE, FILTERS_CONV_3, FILTERS_CONV_4, xavierConv4, ActivationFunction::relu, STRIDE_CONV_4, "ConvLayer4");
  auto conv5 = new ConvLayer(WIDTH_CONV_5, WIDTH_CONV_5, BATCH_SIZE, FILTERS_CONV_4, FILTERS_CONV_5, xavierConv5, ActivationFunction::relu, STRIDE_CONV_5, "ConvLayer5");

  auto maxPool1 = new MaxPool2DLayer(MAX_POOL_1_WIDTH, MAX_POOL_1_HEIGHT, MAX_POOL_1_STRIDE, "MaxPool1");

  auto conv1WeightsCopy = conv1->getWeights().copy();
  auto conv2WeightsCopy = conv2->getWeights().copy();

  computationalGraph.addLayer(conv1);
  computationalGraph.addLayer(conv2);
  computationalGraph.addLayer(conv3);
//  computationalGraph.addLayer(maxPool1);
  computationalGraph.addLayer(conv4);
  computationalGraph.addLayer(conv5);
  computationalGraph.addLayer(new FlattenLayer(BATCH_SIZE));

  auto DENSE_LAYER_UNITS = 128;
  auto xavier1 = new XavierInitializer(SIZE_CONV_5 * SIZE_CONV_5 * FILTERS_CONV_5, DENSE_LAYER_UNITS);
//  auto xavier1 = new XavierInitializer(SIZE_MAX_POOL_1 * SIZE_MAX_POOL_1 * FILTERS_CONV_2, DENSE_LAYER_UNITS);
  auto xavier2 = new XavierInitializer(DENSE_LAYER_UNITS, 10);

  computationalGraph.addDenseLayer(
//    {
//      {"width",     780 * 64 },
//      {"height",    128 },
//      {"batchSize", BATCH_SIZE}
//    },
//    SIZE_CONV_2 * SIZE_CONV_2 * FILTERS_CONV_2,
    SIZE_CONV_5 * SIZE_CONV_5 * FILTERS_CONV_5,
//    SIZE_MAX_POOL_1 * SIZE_MAX_POOL_1 * FILTERS_CONV_3,
//    SIZE_MAX_POOL_1 * SIZE_MAX_POOL_1 * FILTERS_CONV_2,
    DENSE_LAYER_UNITS,
    BATCH_SIZE,
    xavier1,
    ActivationFunction ::relu,
    0.0,
    "DenseLayer1"
  );

  computationalGraph.addDenseLayer(
//    {
//      {"width",     128},
//      {"height",    10},
//      {"batchSize", BATCH_SIZE}
//    },
    DENSE_LAYER_UNITS,
    10,
    BATCH_SIZE,
    xavier2,
    ActivationFunction ::softmax,
    0.0,
    "DenseLayer2"
  );

  free(xavier1);
  free(xavier2);
  free(xavierConv1);
  free(xavierConv2);


//  BaseOptimizer *optimizerPtr;
  auto optimizerPtr = new AdamOptimizer(computationalGraph, all_instances, all_labels, training_indices, BATCH_SIZE, 0.001);
//  auto optimizerPtr = new MomentumOptimizer(computationalGraph, all_instances, all_labels, training_indices, BATCH_SIZE, 0.01);
//  auto optimizerPtr = new MiniBatchOptimizer(*computationalGraphPtr, all_instances, all_labels, training_indices, BATCH_SIZE, 0.001);

  auto &optimizer = *optimizerPtr;

  for (batch = 0; secondsPassed < (60 * 28); batch++) {
    optimizerPtr->train();

    secondsPassed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count() - secondsComputingAccuracy;
//    if (batch % 10 == 0) {
//      std::cout << std::endl << "Processed examples: " << batch * BATCH_SIZE << std::endl;
//    }
  }

  std::cout << "Weights absolute diff conv 1:" << conv1->getWeights().totalAbsDifference(conv1WeightsCopy) << std::endl;
  std::cout << "Weights absolute diff conv 2:" << conv2->getWeights().totalAbsDifference(conv2WeightsCopy) << std::endl;


  std::cout << std::endl << "Processed examples: " << batch * BATCH_SIZE << std::endl;
  std::cout << std::endl << "Validation computing (s): " << secondsComputingAccuracy << std::endl;

  std::cout << "Validation ";
  auto validationAccuracy = computeAccuracy(
    inputs,
    all_instances,
    expectedOutputs,
    all_labels,
    *computationalGraphPtr,
    BATCH_SIZE,
    validation_indices
  )["accuracy"];

//  std::cout << "Train ";
//  auto trainAccuracy = computeAccuracy(
//    inputs,
//    all_instances,
//    expectedOutputs,
//    all_labels,
//    *computationalGraphPtr,
//    BATCH_SIZE,
//    training_indices
//  )["accuracy"];


  std::cout << "Test ";

  auto testAccuracy = computeAccuracy(
    inputs,
    all_test_instances,
    expectedOutputs,
    all_test_labels,
    *computationalGraphPtr,
    BATCH_SIZE,
    test_indices
  )["accuracy"];


  std::map<std::string, double> results = {
    { "validationAccuracy", validationAccuracy },
//    { "trainingAccuracy", trainAccuracy },
    { "testAccuracy", testAccuracy },
  };


  return computationalGraphPtr;
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

  performTraining(
    all_instances,
    all_labels,
    training_indices,
    validation_indices,

    all_test_instances,
    all_test_labels,
    test_indices
  );

  return 0;
}
