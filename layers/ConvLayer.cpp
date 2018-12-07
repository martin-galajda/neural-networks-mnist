//
// Created by Martin Galajda on 27/11/2018.
//

#include "ConvLayer.h"
#include "../matrix_impl/Matrix.hpp"
#include "../ops/Convolution.h"
#include "../initializers/XavierInitializer.h"
#include <chrono>

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

  this->inputs = MatrixDoubleSharedPtr(new MatrixDouble(X->getNumOfRows(), X->getNumOfCols(), X->getDepth(), X->getBatchSize()));
  this->inputs = X;

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
    this->activatedInputs->copyElementsFrom(*layerOutput, false);
  }



  return layerOutput;
}


std::shared_ptr<Matrix<double>> ConvLayer::backPropagate(std::shared_ptr<Matrix<double>> forwardDerivatives) {
  auto &dLossWithRespectToOutput = *forwardDerivatives;

  auto &derivLossWithRespectToX = *neuronDerivatives;
  derivLossWithRespectToX.setAllElementsZero();

  auto &derivLossWithRespectToKernel = *weightsDerivatives;
  derivLossWithRespectToKernel.setAllElementsZero();

  if (activationFunction == ActivationFunction::relu) {
    // inplace = true
    dLossWithRespectToOutput.componentWiseReluDerivMult(this->activatedInputs.get(), true);
  }

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


              if (currentRowPos >= inputs->getNumOfRows()) {
                continue;
              }


              for (auto convolutionColPos = 0; convolutionColPos < dLossWithRespectToOutput.getNumOfCols(); convolutionColPos++) {
                auto currentColPos = kernelColIdx + strideCol;

                if (currentColPos >= inputs->getNumOfCols()) {
                  continue;
                }

                *derivLossWithRespectToKernel(kernelRowIdx, kernelColIdx, kernelDepthIdx, kernelIdx) +=
                  *dLossWithRespectToOutput(convolutionRowPos, convolutionColPos, kernelIdx, sampleIdx)
                  * *(*inputs)(currentRowPos, currentColPos, kernelDepthIdx, sampleIdx);


                *derivLossWithRespectToX(currentRowPos, currentColPos, kernelDepthIdx, sampleIdx) +=
                  *dLossWithRespectToOutput(convolutionRowPos, convolutionColPos, kernelIdx, sampleIdx)
                  * *(*weights)(kernelRowIdx, kernelColIdx, kernelDepthIdx, kernelIdx);

                strideCol += stride;
              }

              strideRow += stride;
            }
          }
        }
      }
    }
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

bool ConvLayer::hasWeights() {
  return true;
}

bool ConvLayer::hasBiases() {
  return false;
}