//
// Created by Martin Galajda on 03/12/2018.
//

#ifndef MATRIXBENCHMARKS_MAXPOOL2DLAYER_H
#define MATRIXBENCHMARKS_MAXPOOL2DLAYER_H

#include "BaseLayer.h"

class MaxPool2DLayer: public BaseLayer {
public:
  MaxPool2DLayer(
    int kernelWidth,
    int kernelHeight,
    int stride,
    std::string name = "MaxPool2DLayer"
  ): BaseLayer(name),  kernelWidth(kernelWidth), kernelHeight(kernelHeight), stride(stride) {}

  virtual std::shared_ptr<Matrix<double>> forwardPropagate(std::shared_ptr<Matrix<double>> X);
  virtual std::shared_ptr<Matrix<double>> activate(std::shared_ptr<Matrix<double>> &X);
  virtual std::shared_ptr<Matrix<double>> backPropagate(std::shared_ptr<Matrix<double>> forwardDerivatives);

  virtual bool hasBiases() { return false; }
  virtual bool hasWeights() { return false; }
protected:
  int kernelWidth;
  int kernelHeight;
  int stride;

  std::shared_ptr<Matrix<double>> rowsFlags;
  std::shared_ptr<Matrix<double>> colsFlags;
};


#endif //MATRIXBENCHMARKS_MAXPOOL2DLAYER_H
