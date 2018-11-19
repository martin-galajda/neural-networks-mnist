//
// Created by Martin Galajda on 28/10/2018.
//

#include "ComputationalGraph.h"

#define NOT_PRINT_BENCHMARK

ComputationalGraph::ComputationalGraph() {
    std::list<std::shared_ptr<DenseLayer>> layers;
    this->layers = layers;
}

void ComputationalGraph::addLayer(std::shared_ptr<DenseLayer> layer) {
    this->layers.push_back(layer);
}

void ComputationalGraph::addLayer(
        std::map<std::string, int> layerSizeDefinition,
        BaseInitializer *initializer,
        ActivationFunction activationFunction,
        double l2regularization
) {

    int layerWidth = layerSizeDefinition["width"];
    int layerHeight = layerSizeDefinition["height"];
    int layerBatchSize = layerSizeDefinition["batchSize"];

    auto newLayer = std::shared_ptr<DenseLayer>(new DenseLayer(layerWidth, layerHeight, layerBatchSize, initializer, activationFunction));

    if (l2regularization != 0.0) {
        newLayer->setL2Regularization(l2regularization);
    }

    this->layers.push_back(newLayer);
}

std::shared_ptr<Matrix<double>> ComputationalGraph::forwardPass(std::shared_ptr<Matrix<double>> input) {
    auto &lastOutput = input;
    for (auto layerIt = this->layers.begin(); layerIt != this->layers.end(); layerIt++) {
        auto &currLayer = *(*layerIt);

        lastOutput = currLayer.forwardPropagate(lastOutput);
    }


    return lastOutput;
}

void ComputationalGraph::backwardPass(std::shared_ptr<Matrix<double>> lossDerivatives){
    std::shared_ptr<Matrix<double>> &lastDerivatives = lossDerivatives;

    for (auto layerIt = this->layers.rbegin(); layerIt != this->layers.rend(); layerIt++) {
        auto &currLayer = *(*layerIt);

        lastDerivatives = currLayer.backPropagate(lastDerivatives);
    }
}

void ComputationalGraph::learn(){
    for (auto layerIt = this->layers.rbegin(); layerIt != this->layers.rend(); layerIt++) {
        auto &currLayer = *(*layerIt);

        currLayer.updateWeights();
    }
}

void ComputationalGraph::learn(double learningRate){

    for (auto layerIt = this->layers.rbegin(); layerIt != this->layers.rend(); layerIt++) {
        auto &currLayer = *(*layerIt);

        currLayer.getWeightsDerivatives() *= learningRate;
        currLayer.getBiasesDerivatives() *= learningRate;

        currLayer.updateWeights();
    }
}