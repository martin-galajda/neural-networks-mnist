//
// Created by Martin Galajda on 28/10/2018.
//

#include <list>
#include <BaseLayer.h>
#include <memory>
#include <map>

#ifndef NEURAL_NETWORKS_COMPUTATIONALGRAPH_H
#define NEURAL_NETWORKS_COMPUTATIONALGRAPH_H


class ComputationalGraph {
public:
    ComputationalGraph(int inputSizeRows, int inputSizeCols, int outputSize);
    ComputationalGraph();

    ComputationalGraph & operator=(const ComputationalGraph&) = delete;
    ComputationalGraph(const ComputationalGraph&) = delete;

    void addLayer(BaseLayer *layer);

    void addDenseLayer(std::map<std::string, int> layerSizeDefinition, BaseInitializer *initializer,
                       ActivationFunction activationFunction, double = 0.0, std::string name = "DenseLayer");
    void addDenseLayer(int, int, int, BaseInitializer *initializer,
                       ActivationFunction activationFunction, double = 0.0, std::string name = "DenseLayer");
    std::shared_ptr<Matrix<double>> forwardPass(std::shared_ptr<Matrix<double>> input);

    MatrixDoubleSharedPtr backwardPass(std::shared_ptr<Matrix<double>> lossDerivatives);

    int getInputSizeRows() { return this->inputSizeRows; }
    int getInputSizeCols() { return this->inputSizeCols; }
    int getOutputSize() { return this->outputSize; }

    void learn();
    void learn(double);

    std::list<BaseLayer *> &getLayers() { return layers; }
protected:
    int inputSizeRows = 0;
    int inputSizeCols = 0;
    int outputSize = 0;
    std::list<BaseLayer *> layers;
};


#endif //NEURAL_NETWORKS_COMPUTATIONALGRAPH_H
