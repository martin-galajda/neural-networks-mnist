//
// Created by Martin Galajda on 28/10/2018.
//
#define _GLIBCXX_USE_CXX11_ABI 0

#include <list>
#include <DenseLayer.h>
#include <memory>
#include <map>

#ifndef MATRIXBENCHMARKS_COMPUTATIONALGRAPH_H
#define MATRIXBENCHMARKS_COMPUTATIONALGRAPH_H


class ComputationalGraph {
public:
    ComputationalGraph();

    ComputationalGraph & operator=(const ComputationalGraph&) = delete;
    ComputationalGraph(const ComputationalGraph&) = delete;

    void addLayer(std::shared_ptr<DenseLayer> layer);

    void addLayer(std::map<std::string, int> layerSizeDefinition, BaseInitializer *initializer, ActivationFunction activationFunction, double = 0.0);
    std::shared_ptr<Matrix<double>> forwardPass(std::shared_ptr<Matrix<double>> input);

    void backwardPass(std::shared_ptr<Matrix<double>> lossDerivatives);

    void learn();
    void learn(double);

    std::list<std::shared_ptr<DenseLayer>> &getLayers() { return layers; }
protected:

    std::list<std::shared_ptr<DenseLayer>> layers;
};


#endif //MATRIXBENCHMARKS_COMPUTATIONALGRAPH_H
