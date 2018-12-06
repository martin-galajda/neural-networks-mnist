//
// Created by Martin Galajda on 05/12/2018.
//

#include "BatchNormalization.h"
#include "cmath"

BatchNormalization::BatchNormalization(int inputRows, std::string name): BaseLayer(name), inputRows(inputRows) {
  this->weights = std::shared_ptr<Matrix<double>> (new MatrixDouble(1, 1, 1, 2));

  auto gammasBeginPtr = (*this->weights)(0,0,0,0);
  auto betasBeginPtr = (*this->weights)(0,0,0,1);

  bool shouldCopyMemory = false;
  this->gammas = std::shared_ptr<Matrix<double>> (new MatrixDouble(gammasBeginPtr, 1, 1, shouldCopyMemory, 1, 1));
  this->betas = std::shared_ptr<Matrix<double>> (new MatrixDouble(betasBeginPtr, 1, 1, shouldCopyMemory, 1, 1));

  this->gammas->setAllElementsTo(1.0);
  this->betas->setAllElementsZero();
}


std::shared_ptr<Matrix<double>> BatchNormalization::forwardPropagate(std::shared_ptr<Matrix<double>> input) {
  auto epsilon = 1e8;
  this->means = std::shared_ptr<Matrix<double>> (new MatrixDouble(1, 1, 1, 1));
  this->means->setAllElementsZero();


  this->variances = std::shared_ptr<Matrix<double>> (new MatrixDouble(1, 1, 1, 1));
  this->variances->setAllElementsZero();


  for (auto batchIdx = 0; batchIdx < input->getBatchSize(); batchIdx++) {
    for (auto depthIdx = 0; depthIdx < input->getDepth(); depthIdx++) {
      for (auto rowIdx = 0; rowIdx < input->getNumOfRows(); rowIdx++) {
        for (auto colIdx = 0; colIdx < input->getNumOfCols(); colIdx++) {
          *(*this->means)(0, 0, 0, 0) += *(*input)(rowIdx, colIdx, depthIdx, batchIdx);
        }
      }
    }
  }

  *(*this->means)(0, 0, 0, 0) /= input->getBatchSize();

  for (auto batchIdx = 0; batchIdx < input->getBatchSize(); batchIdx++) {
    for (auto depthIdx = 0; depthIdx < input->getDepth(); depthIdx++) {
      for (auto rowIdx = 0; rowIdx < input->getNumOfRows(); rowIdx++) {
        for (auto colIdx = 0; colIdx < input->getNumOfCols(); colIdx++) {
          auto currentFeatureValue = *(*input)(rowIdx, colIdx, depthIdx, batchIdx);
          auto mean = *(*this->means)(0, 0, 0, 0);
          *(*this->variances)(0, 0, 0, 0) += (currentFeatureValue - mean);
        }
      }
    }
  }

  *(*this->variances)(0, 0, 0, 0) /= input->getBatchSize();

  this->activatedInputs = std::shared_ptr<Matrix<double>> (input->copy());

  for (auto batchIdx = 0; batchIdx < input->getBatchSize(); batchIdx++) {
    for (auto depthIdx = 0; depthIdx < input->getDepth(); depthIdx++) {
      for (auto rowIdx = 0; rowIdx < input->getNumOfRows(); rowIdx++) {
        for (auto colIdx = 0; colIdx < input->getNumOfCols(); colIdx++) {
          auto currentFeatureValue = *(*input)(rowIdx, colIdx, depthIdx, batchIdx);
          auto mean = *(*this->means)(0, 0, 0, 0);
          auto variance = *(*this->variances)(0, 0, 0, 0);

          auto normalizedFeatureValue = ((currentFeatureValue - mean) / (std::sqrt(variance + epsilon)));

          auto scaled = normalizedFeatureValue * *(*this->gammas)(0,0,0,0);

          auto shifted = scaled + *(*this->betas)(0,0,0,0);

          *(*this->activatedInputs)(rowIdx, colIdx, depthIdx, batchIdx) = shifted;
        }
      }
    }
  }


  return this->activatedInputs;
}

std::shared_ptr<Matrix<double>> BatchNormalization::backPropagate(std::shared_ptr<Matrix<double>> forwardDerivatives) {


}

std::shared_ptr<Matrix<double>> BatchNormalization::activate(std::shared_ptr<Matrix<double>> &X) {
  return X;
}