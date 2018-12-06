
#include "./BaseLayer.h"

void BaseLayer::updateWeights() {
  *this->weights -= *weightsDerivatives;

  if (this->hasBiases()) {
    *this->biases -= *biasesDerivatives;
  }
}

void BaseLayer::setL2Regularization(double decayStrength) {
    this->regularization = Regularization :: l2;
    this->regularizer = std::shared_ptr<L2Regularizer>(new L2Regularizer(this, decayStrength));
}

std::shared_ptr<L2Regularizer> BaseLayer::getRegularizer() {
    return this->regularizer;
}
