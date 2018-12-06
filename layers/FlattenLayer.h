//
// Created by Martin Galajda on 01/12/2018.
//

#ifndef MATRIXBENCHMARKS_FLATTENLAYER_H
#define MATRIXBENCHMARKS_FLATTENLAYER_H

#include "BaseLayer.h"

class FlattenLayer: public BaseLayer {

public:
  FlattenLayer(int batchSize, std::string name = "FlattenLayer");

  virtual std::shared_ptr<Matrix<double>> forwardPropagate(std::shared_ptr<Matrix<double>> X);
  virtual std::shared_ptr<Matrix<double>> activate(std::shared_ptr<Matrix<double>> &X);
  virtual std::shared_ptr<Matrix<double>> backPropagate(std::shared_ptr<Matrix<double>> forwardDerivatives);

  virtual bool hasBiases();
  virtual bool hasWeights();

protected:
  int cachedNumOfRows;
  int cachedNumOfCols;
  int cachedDepth;
};


#endif //MATRIXBENCHMARKS_FLATTENLAYER_H
