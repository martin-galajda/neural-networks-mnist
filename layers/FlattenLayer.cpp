//
// Created by Martin Galajda on 01/12/2018.
//

#include "FlattenLayer.h"


//
// Created by Martin Galajda on 27/11/2018.
//

#include "ConvLayer.h"
#include "../matrix_impl/Matrix.hpp"
#include "../ops/Convolution.h"

FlattenLayer::FlattenLayer(int batchSize, std::string name): BaseLayer(batchSize, name) {}

MatrixDoubleSharedPtr FlattenLayer::forwardPropagate(MatrixDoubleSharedPtr X) {
  this->cachedDepth = X->getDepth();
  this->cachedNumOfRows = X->getNumOfRows();
  this->cachedNumOfCols = X->getNumOfCols();

//  X->reshape(X->getNumOfRows() * X->getNumOfCols() * X->getDepth(), 1, 1,  X->getBatchSize());

  this->activatedInputs = MatrixDoubleSharedPtr(X->copy());

  this->activatedInputs->reshape(X->getNumOfRows() * X->getNumOfCols() * X->getDepth(), 1, 1,  X->getBatchSize());

  return this->activatedInputs;
}

std::shared_ptr<Matrix<double>> FlattenLayer::backPropagate(std::shared_ptr<Matrix<double>> forwardDerivatives, int numOfThreads) {
  forwardDerivatives->reshape(this->cachedNumOfRows, this->cachedNumOfCols, this->cachedDepth, batchSize);
  this->neuronDerivatives = MatrixDoubleSharedPtr(forwardDerivatives->copy());
  this->neuronDerivatives->reshape(this->cachedNumOfRows, this->cachedNumOfCols, this->cachedDepth, batchSize);
  return this->neuronDerivatives;
}

std::shared_ptr<Matrix<double>> FlattenLayer::activate(std::shared_ptr<Matrix<double>> &X) {
  return X;
}
