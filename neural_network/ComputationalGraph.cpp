//
// Created by Martin Galajda on 28/10/2018.
//

#include "ComputationalGraph.h"
#include "../layers/DenseLayer.h"
#include "../layers/ConvLayer.h"

ComputationalGraph::ComputationalGraph() {
    std::vector<BaseLayer *> layers;
    this->layers = layers;
}

ComputationalGraph::ComputationalGraph(int graphInputSizeRows, int graphInputSizeCols, int graphOutputSize): ComputationalGraph() {
    this->inputSizeRows = graphInputSizeRows;
    this->inputSizeCols = graphInputSizeCols;
    this->outputSize = graphOutputSize;
}

void ComputationalGraph::addLayer(BaseLayer *layer) {
    this->layers.push_back(layer);
}

void ComputationalGraph::addDenseLayer(
        std::map<std::string, int> layerSizeDefinition,
        BaseInitializer *initializer,
        ActivationFunction activationFunction,
        double l2regularization,
        std::string name
) {

    int layerWidth = layerSizeDefinition["width"];
    int layerHeight = layerSizeDefinition["height"];
    int layerBatchSize = layerSizeDefinition["batchSize"];

    auto newLayer = new DenseLayer(layerWidth, layerHeight, layerBatchSize, initializer, activationFunction, name);

    if (l2regularization != 0.0) {
        newLayer->setL2Regularization(l2regularization);
    }

    this->layers.push_back(newLayer);
}

void ComputationalGraph::addDenseLayer(
        int layerHeight,
        int layerBatchSize,
        ActivationFunction activationFunction,
        std::string name
) {

    auto newLayer = new DenseLayer(layerHeight, layerBatchSize, activationFunction, name);

    this->layers.push_back(newLayer);
}


void ComputationalGraph::addConvLayer(
        int kernelWidth,
        int kernelHeight,
        int stride,
        int numberOfFilters,
        int batchSize,
        ActivationFunction activationFunction,
        std::string name
) {

    auto isFirstLayer = this->layers.size() == 0;
    auto layerDepth = 1;

    if (!isFirstLayer) {
      auto lastLayer = this->layers.back();

      layerDepth = lastLayer->getLayerOutputDepth();
    }

    auto conv = new ConvLayer(kernelWidth, kernelHeight, batchSize, layerDepth, numberOfFilters, activationFunction, stride, name);

    this->addLayer(conv);
}

std::shared_ptr<Matrix<double>> ComputationalGraph::forwardPass(std::shared_ptr<Matrix<double>> input) {
    auto &lastOutput = input;

    for (auto layerIt = this->layers.begin(); layerIt != this->layers.end(); layerIt++) {
        auto &currLayer = *layerIt;
        lastOutput = currLayer->forwardPropagate(lastOutput);
    }

    return lastOutput;
}

MatrixDoubleSharedPtr ComputationalGraph::backwardPass(std::shared_ptr<Matrix<double>> lossDerivatives, int numOfThreads){
    std::shared_ptr<Matrix<double>> &lastDerivatives = lossDerivatives;

    auto idx = layers.size();
    for (auto layerIt = this->layers.rbegin(); layerIt != this->layers.rend(); layerIt++) {
      auto &currLayer = *layerIt;
      lastDerivatives = currLayer->backPropagate(lastDerivatives, numOfThreads);
    }


    return lastDerivatives;
}

void ComputationalGraph::learn(){
    for (auto layerIt = this->layers.rbegin(); layerIt != this->layers.rend(); layerIt++) {
        auto &currLayer = *(*layerIt);

        if (currLayer.hasWeights()) {
            currLayer.updateWeights();
        }
    }
}

void ComputationalGraph::learn(double learningRate){

    for (auto layerIt = this->layers.rbegin(); layerIt != this->layers.rend(); layerIt++) {
        auto &currLayer = *(*layerIt);

        if (currLayer.hasBiases()) {
            currLayer.getBiasesDerivatives() *= learningRate;
        }

        if (currLayer.hasWeights()) {
            currLayer.getWeightsDerivatives() *= learningRate;
            currLayer.updateWeights();
        }
    }
}
