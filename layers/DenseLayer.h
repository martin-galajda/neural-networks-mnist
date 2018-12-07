//
// Created by Martin Galajda on 26/10/2018.
//

#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "../matrix_impl/Matrix.hpp"
#include "../initializers/NormalInitializer.h"
#include "./L2Regularizer.h"
#include "./BaseLayer.h"
#include <memory>
#include <vector>
#include <memory>

class DenseLayer: public BaseLayer {
public:
  DenseLayer(int width, int height, int batchSize, double *data, ActivationFunction activationFunction);
  DenseLayer(int width, int height, int batchSize, BaseInitializer *initializer, ActivationFunction activationFunction, std::string name = "DenseLayer");

  DenseLayer(int outputUnits, int batchSize, ActivationFunction activationFunction, std::string name);

  DenseLayer & operator=(const DenseLayer&) = delete;
  DenseLayer(const DenseLayer&) = delete;

  ~DenseLayer();

  void initialize(std::shared_ptr<Matrix<double>> X);

  virtual std::shared_ptr<Matrix<double>> forwardPropagate(std::shared_ptr<Matrix<double>> X);
  virtual std::shared_ptr<Matrix<double>> activate(std::shared_ptr<Matrix<double>> &X);
  virtual std::shared_ptr<Matrix<double>> backPropagate(std::shared_ptr<Matrix<double>> forwardDerivatives);

  virtual bool hasBiases();
  virtual bool hasWeights();

protected:
  bool isInitialized = false;
  int height;
};


#endif //DENSELAYER_H
