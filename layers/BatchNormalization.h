//
// Created by Martin Galajda on 05/12/2018.
//

#ifndef NEURAL_NETWORKS_BATCHNORMALIZATION_H
#define NEURAL_NETWORKS_BATCHNORMALIZATION_H

#include "./BaseLayer.h"

class BatchNormalization: public BaseLayer {
public:
  BatchNormalization(std::string name = "BatchNormalization"): BaseLayer(name) {}
  BatchNormalization(int inputRows, std::string name = "BatchNormalization");

  virtual std::shared_ptr<Matrix<double>> forwardPropagate(std::shared_ptr<Matrix<double>> X);
  virtual std::shared_ptr<Matrix<double>> activate(std::shared_ptr<Matrix<double>> &X);
  virtual std::shared_ptr<Matrix<double>> backPropagate(std::shared_ptr<Matrix<double>> forwardDerivatives);

  virtual bool hasBiases() { return false; }
  virtual bool hasWeights() { return true; }
protected:

  int inputRows = 0;
  MatrixDoubleSharedPtr means;
  MatrixDoubleSharedPtr variances;

  MatrixDoubleSharedPtr betas;
  MatrixDoubleSharedPtr gammas;

  MatrixDoubleSharedPtr trainingSetMeans;
  MatrixDoubleSharedPtr trainingSetDeviations;

};


#endif //NEURAL_NETWORKS_BATCHNORMALIZATION_H
