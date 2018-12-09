//
// Created by Martin Galajda on 26/10/2018.
//

#include "DenseLayer.h"
#include <vector>
#include "../matrix_impl/Matrix.hpp"
#include "../initializers/ZeroInitializer.h"
#include "./L2Regularizer.h"
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include "../initializers/XavierInitializer.h"
#include <thread>

namespace DenseLayerUtils {
  void PerformDenseLayerBackpropagation(PerformDenseLayerBackpropPayload *payload){
    Matrix<double> &forwardDerivatives = *payload->outDerivativesPtr;

    Matrix<double> &weightDerivatives = *(payload->weightDerivativesPtr);
    Matrix<double> &neuronDerivatives = *(payload->neuronDerivativesPtr);
    Matrix<double> *inputs = payload->inputsPtr;
    Matrix<double> *activatedInputs = payload->activatedInputsPtr;
    Matrix<double> *weights = payload->weightsPtr;
    Matrix<double> *biasesDerivatives = payload->biasesDerivativesPtr;

    auto activationFunction = *payload->activationFunctionPtr;

    // each column contains derivatives for one training example
    for (int instanceIndex = 0; instanceIndex < forwardDerivatives.getBatchSize(); instanceIndex++) {
      for (int neuronIndex = 0; neuronIndex < (*inputs).getNumOfRows(); neuronIndex++) {
        for (int forwardDerivativeIndex = 0; forwardDerivativeIndex < forwardDerivatives.getNumOfRows(); forwardDerivativeIndex++) {

          auto currForwardDerivativeValue = *forwardDerivatives(forwardDerivativeIndex, 0,0, instanceIndex);
          auto currActivatedInputValue = *(*activatedInputs)(forwardDerivativeIndex, 0,0, instanceIndex);

          if (activationFunction == ActivationFunction::relu) {
            double reluDerivative = 0.0;
            if (currActivatedInputValue > 0) {
              reluDerivative = 1.0;
            }
            // we need to include relu derivative
            *(weightDerivatives)(forwardDerivativeIndex, neuronIndex, 0, 0) +=
              *(*inputs)(neuronIndex,0,0, instanceIndex) * reluDerivative * currForwardDerivativeValue;

            *(neuronDerivatives)(neuronIndex, 0, 0, instanceIndex) +=
              *(*weights)(forwardDerivativeIndex, neuronIndex, 0, 0) * reluDerivative * currForwardDerivativeValue;

            // TODO CHECK!!!
            (*biasesDerivatives)[forwardDerivativeIndex][0] += (currForwardDerivativeValue * reluDerivative) / (*inputs).getNumOfRows();
          } else {
            // TODO: As of now we support only softmax (in the last layer used with cross-entropy) and relu
            // softmax derivative already has derivative of activation function passed
            (weightDerivatives)[forwardDerivativeIndex][neuronIndex] += *(*inputs)(neuronIndex, 0,0,instanceIndex) * currForwardDerivativeValue;
            *(neuronDerivatives)(neuronIndex, 0, 0, instanceIndex) += *(*weights)(forwardDerivativeIndex, neuronIndex, 0, 0) * currForwardDerivativeValue;

            // TODO CHECK!!!
            (*biasesDerivatives)[forwardDerivativeIndex][0] += (currForwardDerivativeValue) / (*inputs).getNumOfRows();
          }
        }
      }
    };
  }
}



DenseLayer::~DenseLayer() {}

DenseLayer::DenseLayer(int width, int height, int batchSize, BaseInitializer *initializer, ActivationFunction activationFunction, std::string name): BaseLayer(batchSize, name) {
    this->weights = std::shared_ptr<Matrix<double>>(new Matrix<double>(height, width, initializer));
    this->inputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(width, 1, 1, batchSize));
    this->activatedInputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(height, 1,1, batchSize));
    this->activationFunction = activationFunction;
    this->batchSize = batchSize;

    auto zeroInitializer = new ZeroInitializer();

    this->weightsDerivatives = std::shared_ptr<Matrix<double> >(new Matrix<double>(height, width));
    this->neuronDerivatives = std::shared_ptr<Matrix<double> >(new Matrix<double>(width,1,1, batchSize));
    this->biases = std::shared_ptr<Matrix<double> >(new Matrix<double>(height, 1, zeroInitializer));
    this->biasesDerivatives = std::shared_ptr<Matrix<double> >(new Matrix<double>(height, 1));

    free(zeroInitializer);

    this->isInitialized = true;
}

DenseLayer::DenseLayer(int height, int batchSize, ActivationFunction activationFunction, std::string name): BaseLayer(batchSize, name) {
    this->activationFunction = activationFunction;
    this->batchSize = batchSize;
    this->height = height;
}

void DenseLayer::initialize(std::shared_ptr<Matrix<double>> X) {
  auto inputUnits = X->getNumOfRows();
  auto outputUnits = this->height;
  auto xavier = new XavierInitializer(inputUnits, outputUnits);

  this->weights = std::shared_ptr<Matrix<double>>(new Matrix<double>(outputUnits, inputUnits, xavier));
  this->inputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(inputUnits, 1, 1, batchSize));
  this->activatedInputs = std::shared_ptr<Matrix<double>>(new Matrix<double>(outputUnits, 1, 1, batchSize));

  auto zeroInitializer = new ZeroInitializer();

  this->weightsDerivatives = std::shared_ptr<Matrix<double> >(new Matrix<double>(outputUnits, inputUnits, 1, 1));
  this->neuronDerivatives = std::shared_ptr<Matrix<double> >(new Matrix<double>(inputUnits, 1, 1, batchSize));
  this->biases = std::shared_ptr<Matrix<double> >(new Matrix<double>(outputUnits, 1, zeroInitializer));
  this->biasesDerivatives = std::shared_ptr<Matrix<double> >(new Matrix<double>(outputUnits, 1));

  free(zeroInitializer);
  free(xavier);

  this->isInitialized = true;
}

DenseLayer::DenseLayer(int width, int height, int batchSize, double *data, ActivationFunction activationFunction): BaseLayer(batchSize) {
  this->weights = std::shared_ptr<Matrix<double>>(new Matrix<double>(data, height, width));
  this->activationFunction = activationFunction;
  this->batchSize = batchSize;

  this->weightsDerivatives = std::shared_ptr<Matrix<double> >(new Matrix<double>(height, width));
  this->neuronDerivatives = std::shared_ptr<Matrix<double> >(new Matrix<double>(width, 1, 1, batchSize));

  this->isInitialized = true;
}

std::shared_ptr<Matrix<double>> DenseLayer::forwardPropagate(std::shared_ptr<Matrix<double>> X){
  if (!this->isInitialized) {
    this->initialize(X);
  }
  auto t_start = std::chrono::high_resolution_clock::now();

  this->inputs = X;

  auto A = std::shared_ptr<Matrix<double>>((*this->weights) * (*X));

  for (auto i = 0; i < A->getNumOfRows(); i++) {
      for (auto j = 0; j < A ->getBatchSize(); j++) {
          *(*A)(i, 0, 0, j) += (*biases)[i][0];
      }
  }


  auto activatedInputs = this->activate(A);
  this->activatedInputs = activatedInputs;

  double secondsPassed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();

//  std::cout << this->name << "computing forward prop: " << secondsPassed << std::endl;

  return activatedInputs;
}

std::shared_ptr<Matrix<double>> DenseLayer::backPropagate(std::shared_ptr<Matrix<double>> forwardDerivativesPtr, int numOfThreads) {
  auto t_start = std::chrono::high_resolution_clock::now();

  this->neuronDerivatives->reshape(inputs->getNumOfRows(),1,1, batchSize);
  Matrix<double> &forwardDerivatives = *forwardDerivativesPtr;

  if (forwardDerivatives.getBatchSize() != this->batchSize) {
      std::stringstream errorString;
      errorString << "Got invalid number of columns for forward derivatives. Expected:  "
          << this->batchSize
          << " but I got "
          << forwardDerivatives.getBatchSize()
          << std::endl;

      throw std::invalid_argument(errorString.str());
  }

  Matrix<double> &weightDerivatives = *(this->weightsDerivatives);
  Matrix<double> &neuronDerivatives = *(this->neuronDerivatives);

  weightDerivatives.setAllElementsZero();
  biasesDerivatives->setAllElementsZero();
  neuronDerivatives.setAllElementsZero();


  std::thread *threads[numOfThreads];
  struct DenseLayerUtils::PerformDenseLayerBackpropPayload dataForJob[numOfThreads];

  // integer divison with ceiling up ...
  auto threadBatchSize = batchSize % numOfThreads ? (batchSize / numOfThreads) + 1 : batchSize / numOfThreads;

  auto outRows = forwardDerivatives.getNumOfRows();
  auto outCols = forwardDerivatives.getNumOfCols();
  auto outDepth = forwardDerivatives.getDepth();

  auto inRows = inputs->getNumOfRows();
  auto inCols = inputs->getNumOfCols();
  auto inDepth = inputs->getDepth();

  auto samplesToProcess = batchSize;

  for (int i = 0; i < numOfThreads; i++) {
    auto jobBatchSize = std::min(samplesToProcess, threadBatchSize);

    auto jobOutDerivatives = new MatrixDouble((forwardDerivatives)(0, 0, 0, threadBatchSize * i), outRows, outCols, false, outDepth, jobBatchSize);
    auto jobActivatedInputsPtr = new MatrixDouble((*activatedInputs)(0, 0, 0, threadBatchSize * i), outRows, outCols, false, outDepth, jobBatchSize);

    auto jobInputsPtr = new MatrixDouble((*inputs)(0, 0, 0, threadBatchSize * i), inRows, inCols, false, inDepth, jobBatchSize);
    auto jobInputDerivatives = new MatrixDouble((neuronDerivatives)(0, 0, 0, threadBatchSize * i), inRows, inCols, false, inDepth, jobBatchSize);

    dataForJob[i].threadId = i;

    dataForJob[i].activatedInputsPtr = jobActivatedInputsPtr;
    dataForJob[i].inputsPtr = jobInputsPtr;
    dataForJob[i].outDerivativesPtr = jobOutDerivatives;
    dataForJob[i].neuronDerivativesPtr = jobInputDerivatives;
    dataForJob[i].activationFunctionPtr = &activationFunction;

    auto jobWeightsDerivatives = weightsDerivatives.get();
    auto jobWeightsPtr = weights.get();
    dataForJob[i].weightsPtr = jobWeightsPtr;
    dataForJob[i].weightDerivativesPtr = jobWeightsDerivatives;

    auto biasesDerivatives = this->biasesDerivatives.get();
    dataForJob[i].biasesDerivativesPtr = biasesDerivatives;

    threads[i] = new std::thread(DenseLayerUtils::PerformDenseLayerBackpropagation, &dataForJob[i]);

    samplesToProcess -= jobBatchSize;
  }

  for (int i = 0; i < numOfThreads; i++ ) {
    threads[i]->join();

    free(threads[i]);
  }

  return this->neuronDerivatives;
}

std::shared_ptr<Matrix<double>> DenseLayer::activate(std::shared_ptr<Matrix<double>> &A) {
    if (this->activationFunction == ActivationFunction::softmax) {
        return std::shared_ptr<Matrix<double>>(A->softmax());
    } else if (this->activationFunction == ActivationFunction::relu) {
        A->reluInPlace();
        return A;
    }

    return nullptr;
}

bool DenseLayer::hasBiases() {
    return true;
}

bool DenseLayer::hasWeights() {
    return true;
}
