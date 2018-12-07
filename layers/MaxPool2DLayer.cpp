//
// Created by Martin Galajda on 03/12/2018.
//

#include "MaxPool2DLayer.h"

std::shared_ptr<Matrix<double>> MaxPool2DLayer::forwardPropagate(std::shared_ptr<Matrix<double>> input) {
  this->inputs = std::shared_ptr<Matrix<double>> (input->copy());

  auto outputWidth = ((input->getNumOfCols() - kernelWidth) / stride) + 1;
  auto outputHeight = ((input->getNumOfRows() - kernelHeight) / stride) + 1;

  auto output = MatrixDoubleSharedPtr(new MatrixDouble(outputHeight, outputWidth, input->getDepth(), input->getBatchSize()));

  auto rowsFlags = MatrixDoubleSharedPtr(new MatrixDouble(outputHeight, outputWidth, input->getDepth(), input->getBatchSize()));
  auto colsFlags = MatrixDoubleSharedPtr(new MatrixDouble(outputHeight, outputWidth, input->getDepth(), input->getBatchSize()));

  this->rowsFlags = rowsFlags;
  this->colsFlags = colsFlags;

  for (auto batchIdx = 0; batchIdx < input->getBatchSize(); batchIdx++) {
    for (auto channelIdx = 0; channelIdx < input->getDepth(); channelIdx++) {
      auto inputIdxStartRow = 0;

      for (auto i = 0; i < output->getNumOfRows(); i++) {
        //do magic here

        auto inputIdxStartCol = 0;
        for (auto j = 0; j < output->getNumOfCols(); j++) {

          if ((input->getNumOfRows() < (inputIdxStartRow + kernelHeight)) || (input->getNumOfCols() < (inputIdxStartCol + kernelWidth))) {
            continue;
          }

          auto max = *(*input)(inputIdxStartRow, inputIdxStartCol, channelIdx, batchIdx);
          auto idxMaxRow = inputIdxStartRow;
          auto idxMaxCol = inputIdxStartCol;
          for (auto kernelRow = 0; kernelRow < kernelHeight; kernelRow++) {
            for (auto kernelCol = 0; kernelCol < kernelWidth; kernelCol++) {
              if ((*(*input)(inputIdxStartRow + kernelRow, inputIdxStartCol + kernelCol, channelIdx, batchIdx)) > max) {
                max = *(*input)(inputIdxStartRow + kernelRow, inputIdxStartCol + kernelCol, channelIdx, batchIdx);
                idxMaxRow = inputIdxStartRow + kernelRow;
                idxMaxCol = inputIdxStartCol + kernelCol;
              }
            }
          }

          *(*output)(i,j,channelIdx,batchIdx) = max;
          *(*rowsFlags)(i, j, channelIdx, batchIdx) = idxMaxRow;
          *(*colsFlags)(i, j, channelIdx, batchIdx) = idxMaxCol;

          // move pointer
          inputIdxStartCol += stride;
        }

        // move pointer
        inputIdxStartRow += stride;
      }
    }
  }


  return output;
}

std::shared_ptr<Matrix<double>> MaxPool2DLayer::backPropagate(std::shared_ptr<Matrix<double>> forwardDerivatives) {
  this->inputs->setAllElementsZero();

  for (auto b = 0; b < forwardDerivatives->getBatchSize(); b++) {
    for (auto d = 0; d < forwardDerivatives->getDepth(); d++) {
      for (auto i = 0; i < forwardDerivatives->getNumOfRows(); i++) {
        for (auto j = 0; j < forwardDerivatives->getNumOfCols(); j++) {
          auto rowIdx = *(*this->rowsFlags)(i,j,d,b);
          auto colIdx = *(*this->colsFlags)(i,j,d,b);

          *(*this->inputs)(rowIdx, colIdx, d, b) = *(*forwardDerivatives)(i,j,d,b);
        }
      }
    }

  }

  return this->inputs;
}

std::shared_ptr<Matrix<double>> MaxPool2DLayer::activate(std::shared_ptr<Matrix<double>> &X) {
  return X;
}
