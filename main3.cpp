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

  auto STRIDE_CONV_1 = 1;
  auto STRIDE_CONV_2 = 1;

  auto WIDTH_CONV_1 = 3;
  auto WIDTH_CONV_2 = 3;

  auto FILTERS_CONV_1 = 2;
  auto FILTERS_CONV_2 = 4;

  computationalGraph.addConvLayer(WIDTH_CONV_1, WIDTH_CONV_1, STRIDE_CONV_1, FILTERS_CONV_1, BATCH_SIZE, ActivationFunction::relu, "ConvLayer1");
  computationalGraph.addConvLayer(WIDTH_CONV_2, WIDTH_CONV_2, STRIDE_CONV_2, FILTERS_CONV_2, BATCH_SIZE, ActivationFunction::relu, "ConvLayer2");

  auto MAX_POOL_1_WIDTH = 2;
  auto MAX_POOL_1_HEIGHT = 2;
  auto MAX_POOL_1_STRIDE = 2;
  auto maxPool1 = new MaxPool2DLayer(MAX_POOL_1_WIDTH, MAX_POOL_1_HEIGHT, MAX_POOL_1_STRIDE, "MaxPool1");
  computationalGraph.addLayer(maxPool1);

  computationalGraph.addLayer(new FlattenLayer(BATCH_SIZE));

  auto DENSE_LAYER_UNITS = 128;
  computationalGraph.addDenseLayer(DENSE_LAYER_UNITS, BATCH_SIZE, ActivationFunction::relu, "DenseLayer1");
  computationalGraph.addDenseLayer(10, BATCH_SIZE, ActivationFunction::softmax, "OutputLayer");


//  BaseOptimizer *optimizerPtr;
  auto optimizerPtr = new AdamOptimizer(computationalGraph, all_instances, all_labels, training_indices, BATCH_SIZE, 0.001);
//  auto optimizerPtr = new MomentumOptimizer(computationalGraph, all_instances, all_labels, training_indices, BATCH_SIZE, 0.01);
//  auto optimizerPtr = new MiniBatchOptimizer(*computationalGraphPtr, all_instances, all_labels, training_indices, BATCH_SIZE, 0.001);

  auto &optimizer = *optimizerPtr;

  for (batch = 0; secondsPassed < (60 * 28); batch++) {
    optimizerPtr->train();
  }


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
