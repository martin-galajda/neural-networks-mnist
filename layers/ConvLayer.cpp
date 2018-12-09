//
// Created by Martin Galajda on 27/11/2018.
//

#include "ConvLayer.h"
#include "../matrix_impl/Matrix.hpp"
#include "../ops/Convolution.h"
#include "../initializers/XavierInitializer.h"
#include <chrono>
#include <thread>

namespace ConvLayerUtils {
  void PerformConvLayerBackPropagation(PerformConvLayerBackpropPayload *payload) {
    auto &dLossWithRespectToOutput = *payload->outDerivativesPtr;

    auto &derivLossWithRespectToX = *payload->neuronDerivativesPtr;
    auto &derivLossWithRespectToKernel = *payload->weightDerivativesPtr;

    auto activationFunction = *payload->activationFunctionPtr;
    auto stride = *payload->stridePtr;
    if (activationFunction == ActivationFunction::relu) {
      // inplace = true
      dLossWithRespectToOutput.componentWiseReluDerivMult(payload->activateInputsPtr, true);
    }

    auto batchSize = payload->outDerivativesPtr->getBatchSize();

    auto inputDepth = payload->inputsPtr->getDepth();

    auto numberOfFilters = payload->weightsPtr->getBatchSize();
    auto kernelWidth = payload->weightsPtr->getNumOfCols();
    auto kernelHeight = payload->weightsPtr->getNumOfRows();

    auto inputRows = payload->inputsPtr->getNumOfRows();
    auto inputCols = payload->inputsPtr->getNumOfCols();

    auto &inputs  = *payload->inputsPtr;
    auto &weights = *payload->weightsPtr;

    // compute derivatives of kernels ... uff, can it be simplified 🤔?
    for (auto sampleIdx = 0; sampleIdx < batchSize; sampleIdx++) {
      for (auto kernelIdx = 0; kernelIdx < numberOfFilters; kernelIdx++) {
        for (auto kernelRowIdx = 0; kernelRowIdx < kernelHeight; kernelRowIdx++) {
          for (auto kernelColIdx = 0; kernelColIdx < kernelWidth; kernelColIdx++) {
            for (auto kernelDepthIdx = 0; kernelDepthIdx < inputDepth; kernelDepthIdx++) {
              auto strideRow = 0;
              for (auto convolutionRowPos = 0; convolutionRowPos < dLossWithRespectToOutput.getNumOfRows(); convolutionRowPos++) {
                auto strideCol = 0;
                auto currentRowPos = kernelRowIdx + strideRow;


                if (currentRowPos >= inputRows) {
                  continue;
                }


                for (auto convolutionColPos = 0; convolutionColPos < dLossWithRespectToOutput.getNumOfCols(); convolutionColPos++) {
                  auto currentColPos = kernelColIdx + strideCol;

                  if (currentColPos >= inputCols) {
                    continue;
                  }

                  *derivLossWithRespectToKernel(kernelRowIdx, kernelColIdx, kernelDepthIdx, kernelIdx) +=
                    *dLossWithRespectToOutput(convolutionRowPos, convolutionColPos, kernelIdx, sampleIdx)
                    * *(inputs)(currentRowPos, currentColPos, kernelDepthIdx, sampleIdx);


                  *derivLossWithRespectToX(currentRowPos, currentColPos, kernelDepthIdx, sampleIdx) +=
                    *dLossWithRespectToOutput(convolutionRowPos, convolutionColPos, kernelIdx, sampleIdx)
                    * *(weights)(kernelRowIdx, kernelColIdx, kernelDepthIdx, kernelIdx);

                  strideCol += stride;
                }

                strideRow += stride;
              }
            }
          }
        }
      }
    }

//  std::cout << "Finishing thread: " << job_data->thread_id << std::endl;

    free(payload->neuronDerivativesPtr);
    free(payload->outDerivativesPtr);
    free(payload->inputsPtr);
    free(payload->activateInputsPtr);
  }
}

ConvLayer::ConvLayer(
  int kernelWidth,
  int kernelHeight,
  int batchSize,
  int inputDepth,
  int numberOfFilters,
  BaseInitializer *initializer,
  ActivationFunction activationFunction,
  int stride,
  std::string name
): BaseLayer(batchSize, name), kernelWidth(kernelWidth), kernelHeight(kernelHeight), inputDepth(inputDepth), numberOfFilters(numberOfFilters) {
  // kernels
  auto kernelCols = kernelWidth;
  auto kernelRows = kernelHeight;

  this->weights = MatrixDoubleSharedPtr(new MatrixDouble(kernelRows, kernelCols, inputDepth, numberOfFilters, initializer));
  this->weightsDerivatives = MatrixDoubleSharedPtr(new MatrixDouble(kernelRows, kernelCols, inputDepth, numberOfFilters));

  this->activationFunction = activationFunction;
  this->batchSize = batchSize;
  this->stride = stride;

  this->isInitialized = true;
}

ConvLayer::ConvLayer(
  int kernelWidth,
  int kernelHeight,
  int batchSize,
  int inputDepth,
  int numberOfFilters,
  ActivationFunction activationFunction,
  int stride,
  std::string name
): BaseLayer(batchSize, name), kernelWidth(kernelWidth), kernelHeight(kernelHeight), inputDepth(inputDepth), numberOfFilters(numberOfFilters) {
  // kernels
  auto kernelCols = kernelWidth;
  auto kernelRows = kernelHeight;

  this->weights = MatrixDoubleSharedPtr(new MatrixDouble(kernelRows, kernelCols, inputDepth, numberOfFilters));
  this->weightsDerivatives = MatrixDoubleSharedPtr(new MatrixDouble(kernelRows, kernelCols, inputDepth, numberOfFilters));

  this->activationFunction = activationFunction;
  this->batchSize = batchSize;
  this->stride = stride;
  this->numberOfFilters = numberOfFilters;
}


MatrixDoubleSharedPtr ConvLayer::forwardPropagate(MatrixDoubleSharedPtr X) {
  if (!this->isInitialized){
    this->initialize(X);
  }
//  this->inputs = X;
  this->inputs = MatrixDoubleSharedPtr(X->copy());

  this->neuronDerivatives = MatrixDoubleSharedPtr(new MatrixDouble(X->getNumOfRows(), X->getNumOfCols(), X->getDepth(), X->getBatchSize()));

  auto outputWidth = ((X->getNumOfCols() - this->kernelWidth) / stride) + 1;
  auto outputHeight = ((X->getNumOfRows() - this->kernelHeight) / stride) + 1;

  auto layerOutput = MatrixDoubleSharedPtr(new MatrixDouble(outputHeight, outputWidth, numberOfFilters, batchSize));
  layerOutput->setAllElementsZero();

  for (int filterIdx = 0; filterIdx < numberOfFilters; filterIdx++) {
    auto kernel = new MatrixDouble((*this->weights)(0, 0, 0, filterIdx), kernelHeight, kernelWidth, false, inputDepth);

    for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++) {
      auto output = new MatrixDouble((*layerOutput)(0, 0, filterIdx, sampleIdx), outputHeight, outputWidth, false);
      auto currentX = MatrixDoubleSharedPtr(new MatrixDouble((*X)(0, 0, 0, sampleIdx), X->getNumOfRows(), X->getNumOfCols(), false, X->getDepth(), 1));

      convolution(currentX, output, kernel, this->stride);


      free(output);
    }

    free(kernel);
  }

  if (activationFunction == ActivationFunction::relu) {
    this->activatedInputs = MatrixDoubleSharedPtr(new MatrixDouble(layerOutput->getNumOfRows(), layerOutput->getNumOfCols(), layerOutput->getDepth(), layerOutput->getBatchSize()));

    layerOutput->reluInPlace();

    // copy memory = true
//    this->activatedInputs->copyElementsFrom(*layerOutput, false);
    this->activatedInputs->copyElementsFrom(*layerOutput, true);
  }

  return layerOutput;
}


std::shared_ptr<Matrix<double>> ConvLayer::backPropagate(std::shared_ptr<Matrix<double>> forwardDerivatives, int numOfThreads) {
  auto t_start = std::chrono::high_resolution_clock::now();

  neuronDerivatives->setAllElementsZero();
  weightsDerivatives->setAllElementsZero();

  std::thread *threads[numOfThreads];

  struct ConvLayerUtils::PerformConvLayerBackpropPayload threadJobData[numOfThreads];

  // integer divison with ceiling up ...
  auto threadBatchSize = batchSize % numOfThreads ? (batchSize / numOfThreads) + 1 : batchSize / numOfThreads;

  auto outRows = forwardDerivatives->getNumOfRows();
  auto outCols = forwardDerivatives->getNumOfCols();
  auto outDepth = forwardDerivatives->getDepth();

  auto inRows = inputs->getNumOfRows();
  auto inCols = inputs->getNumOfCols();
  auto inDepth = inputs->getDepth();

  auto samplesToProcess = batchSize;

  for (int i = 0; i < numOfThreads; i++) {
    auto jobBatchSize = std::min(samplesToProcess, threadBatchSize);

    auto jobOutDerivatives = new MatrixDouble((*forwardDerivatives)(0, 0, 0, threadBatchSize * i), outRows, outCols, false, outDepth, jobBatchSize);
    auto jobActivatedInputsPtr = new MatrixDouble((*activatedInputs)(0, 0, 0, threadBatchSize * i), outRows, outCols, false, outDepth, jobBatchSize);

    auto jobInputsPtr = new MatrixDouble((*inputs)(0, 0, 0, threadBatchSize * i), inRows, inCols, false, inDepth, jobBatchSize);
    auto jobInputDerivatives = new MatrixDouble((*neuronDerivatives)(0, 0, 0, threadBatchSize * i), inRows, inCols, false, inDepth, jobBatchSize);

    threadJobData[i].thread_id = i;
    threadJobData[i].stridePtr = &stride;

    threadJobData[i].activateInputsPtr = jobActivatedInputsPtr;
    threadJobData[i].inputsPtr = jobInputsPtr;
    threadJobData[i].outDerivativesPtr = jobOutDerivatives;
    threadJobData[i].neuronDerivativesPtr = jobInputDerivatives;
    threadJobData[i].activationFunctionPtr = &activationFunction;

    auto jobWeightsDerivatives = weightsDerivatives.get();
    auto jobWeightsPtr = weights.get();
    threadJobData[i].weightsPtr = jobWeightsPtr;
    threadJobData[i].weightDerivativesPtr = jobWeightsDerivatives;

    threads[i] = new std::thread(ConvLayerUtils::PerformConvLayerBackPropagation, &threadJobData[i]);

    samplesToProcess -= jobBatchSize;
  }

  for (int i = 0; i < numOfThreads; i++ ) {
    threads[i]->join();

    free(threads[i]);
  }


  return neuronDerivatives;
}

void ConvLayer::initialize(std::shared_ptr<Matrix<double>> X) {
  auto SIZE_WIDTH_CONV = (int) (((X->getNumOfCols() - kernelWidth) / stride) + 1);
  auto SIZE_HEIGHT_CONV = (int) (((X->getNumOfRows() - kernelHeight) / stride) + 1);

  auto xavier = new XavierInitializer(X->getNumOfCols() * X->getNumOfRows() * X->getDepth(), SIZE_HEIGHT_CONV * SIZE_WIDTH_CONV * numberOfFilters);

  this->weights->initialize(xavier);

  free(xavier);

  this->isInitialized = true;
}

std::shared_ptr<Matrix<double>> ConvLayer::activate(std::shared_ptr<Matrix<double>> &X) {
  return X;
}
