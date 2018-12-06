//
// Created by Martin Galajda on 27/10/2018.
//

#include "gtest/gtest.h"
#include "../matrix_impl/Matrix.hpp"
#include "../layers/DenseLayer.h"
#include "Matrix.cpp"
#include <cmath>
#include "../neural_network/ComputationalGraph.h"
#include "../neural_network/MiniBatchOptimizer.h"
#include "../neural_network/MomentumOptimizer.h"
#include "../neural_network/AdamOptimizer.h"


TEST(denseLayerTests, xorProblem)
{

    const int L1_NUM_OF_NEURONS = 32;
    const int L2_NUM_OF_NEURONS = 64;
    const int BATCH_SIZE = 4;
    const int OUTPUT_CLASSES = 2;
    const int INPUT_DIMENSIONS = 2;

    auto normalInitializer = new NormalInitializer();
    auto normalInitializer2 = new NormalInitializer();
    auto normalInitializer3 = new NormalInitializer();
    auto *l1 = new DenseLayer(
            INPUT_DIMENSIONS /* width */,
            L1_NUM_OF_NEURONS /* height */,
            BATCH_SIZE /* batch size */,
            normalInitializer,
            ActivationFunction::relu
    );
    auto *l2 = new DenseLayer(
            L1_NUM_OF_NEURONS /* width */,
            L2_NUM_OF_NEURONS /* height */,
            BATCH_SIZE /* batch size */,
            normalInitializer2,
            ActivationFunction::relu
    );
    auto *l3 = new DenseLayer(
            L2_NUM_OF_NEURONS /* width */,
            OUTPUT_CLASSES /* height */,
            BATCH_SIZE /* batch size */,
            normalInitializer3,
            ActivationFunction::softmax
    );

    double inputs[] = {
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 1.0
    };

    double y[] = {
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 1.0, 0.0
    };

    auto inputsMatrix = std::shared_ptr<Matrix<double>>(new Matrix<double>(inputs, INPUT_DIMENSIONS, BATCH_SIZE));
    auto outputMatrix = std::shared_ptr<Matrix<double>>(new Matrix<double>(y, OUTPUT_CLASSES, BATCH_SIZE));

    double learningRate = 0.01;
    for (auto i = 0; i < 100000; i++) {
        auto z1 = l1->forwardPropagate(inputsMatrix);
        auto z2 = l2->forwardPropagate(z1);

        auto s = l3->forwardPropagate(z2);

        auto crossEntropyDerivatives = std::shared_ptr<Matrix<double>>(*s - *outputMatrix);
//        s->updateDebug();
//        crossEntropyDerivatives->updateDebug();

        auto l3Derivatives = l3->backPropagate(crossEntropyDerivatives);
        auto l2Derivatives = l2->backPropagate(l3Derivatives);
        l1->backPropagate(l2Derivatives);

        // TODO: update weights manually or define anotehr function...
        l3->updateWeights();
        l2->updateWeights();
        l1->updateWeights();

        if (i % 1000 == 0) {
            std::cout << "Cross entropy loss: "
                      << s->crossEntropyLoss(outputMatrix.get())
                      << std::endl;
        }

    }

    auto z1 = l1->forwardPropagate(inputsMatrix);
    auto z2 = l2->forwardPropagate(z1);
    auto s = l3->forwardPropagate(z2);

    s->printValues();

    free(normalInitializer);
    free(normalInitializer2);
    free(normalInitializer3);
}

TEST(denseLayerTests, xorProblem2)
{

    const int L1_NUM_OF_NEURONS = 32;
    const int BATCH_SIZE = 4;
    const int OUTPUT_CLASSES = 2;
    const int INPUT_DIMENSIONS = 2;

    auto normalInitializer = new NormalInitializer();
    auto normalInitializer2 = new NormalInitializer();
    auto normalInitializer3 = new NormalInitializer();
    auto *l1 = new DenseLayer(
            INPUT_DIMENSIONS /* width */,
            L1_NUM_OF_NEURONS /* height */,
            BATCH_SIZE /* batch size */,
            normalInitializer,
            ActivationFunction::relu
    );
    auto *l2 = new DenseLayer(
            L1_NUM_OF_NEURONS /* width */,
            OUTPUT_CLASSES /* height */,
            BATCH_SIZE /* batch size */,
            normalInitializer3,
            ActivationFunction::softmax
    );

    double inputs[] = {
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 1.0
    };

    double y[] = {
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 1.0, 0.0
    };

    auto inputsMatrix = std::shared_ptr<Matrix<double>>(new Matrix<double>(inputs, INPUT_DIMENSIONS, BATCH_SIZE));
    auto outputMatrix = std::shared_ptr<Matrix<double>>(new Matrix<double>(y, OUTPUT_CLASSES, BATCH_SIZE));

    double learningRate = 0.01;
    for (auto i = 0; i < 1000000; i++) {
        auto z1 = l1->forwardPropagate(inputsMatrix);
        auto s = l2->forwardPropagate(z1);

        auto crossEntropyDerivatives = std::shared_ptr<Matrix<double>>(*s - *outputMatrix);
        *crossEntropyDerivatives /= BATCH_SIZE;

        s->updateDebug();
        crossEntropyDerivatives->updateDebug();

        auto l2Derivatives = l2->backPropagate(crossEntropyDerivatives);
        l1->backPropagate(l2Derivatives);

        // TODO: update weights manually or define anotehr function...

        l2->updateWeights();
        l1->updateWeights();

        if (i % 1000 == 0) {
            std::cout << "Cross entropy loss: "
                      << s->crossEntropyLoss(outputMatrix.get())
                      << std::endl;
        }
    }

    auto z1 = l1->forwardPropagate(inputsMatrix);
    auto s = l2->forwardPropagate(z1);
    s->printValues();

    free(normalInitializer);
    free(normalInitializer2);
    free(normalInitializer3);
}

TEST(denseLayerTests, xorProblemWithComputationalGraph)
{

    const int L1_NUM_OF_NEURONS = 32;
    const int BATCH_SIZE = 4;
    const int OUTPUT_CLASSES = 2;
    const int INPUT_DIMENSIONS = 2;

    auto normalInitializer = new NormalInitializer();
    auto normalInitializer2 = new NormalInitializer();

    double inputs[] = {
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 1.0
    };

    double y[] = {
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 1.0, 0.0
    };

    auto inputsMatrix = std::shared_ptr<Matrix<double>>(new Matrix<double>(inputs, INPUT_DIMENSIONS, BATCH_SIZE));
    auto outputMatrix = std::shared_ptr<Matrix<double>>(new Matrix<double>(y, OUTPUT_CLASSES, BATCH_SIZE));

    double learningRate = 0.01;

    ComputationalGraph computationalGraph;

    // First layer - hidden
    computationalGraph.addDenseLayer(
            {
                    {"width",     INPUT_DIMENSIONS},
                    {"height",    L1_NUM_OF_NEURONS},
                    {"batchSize", BATCH_SIZE}
            },
            normalInitializer,
            ActivationFunction::relu);

    // Second layer - output
    computationalGraph.addDenseLayer(
            {
                    {"width",     L1_NUM_OF_NEURONS},
                    {"height",    OUTPUT_CLASSES},
                    {"batchSize", BATCH_SIZE}
            },
            normalInitializer2,
            ActivationFunction::softmax);


    for (auto i = 0; i < 10000; i++) {
        auto s = computationalGraph.forwardPass(inputsMatrix);

        s->updateDebug();
        auto crossEntropyDerivatives = std::shared_ptr<Matrix<double>>(*s - *outputMatrix);
        *crossEntropyDerivatives /= BATCH_SIZE;

        computationalGraph.backwardPass(crossEntropyDerivatives);

        // TODO: update weights manually or define anotehr function...
        computationalGraph.learn(learningRate);
    }

    auto s = computationalGraph.forwardPass(inputsMatrix);
    s->printValues();

    free(normalInitializer);
    free(normalInitializer2);
}



TEST(denseLayerTests, xorProblemWithOptimizer)
{

    const int L1_NUM_OF_NEURONS = 32;
    int BATCH_SIZE = 4;
    const int OUTPUT_CLASSES = 2;
    const int INPUT_DIMENSIONS = 2;

    auto normalInitializer = new NormalInitializer();
    auto normalInitializer2 = new NormalInitializer();

    double inputs[] = {
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 1.0
    };

    double y[] = {
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 1.0, 0.0
    };

    auto inputsMatrix = std::shared_ptr<Matrix<double>>(new Matrix<double>(inputs, INPUT_DIMENSIONS, BATCH_SIZE));
    auto outputMatrix = std::shared_ptr<Matrix<double>>(new Matrix<double>(y, OUTPUT_CLASSES, BATCH_SIZE));

    auto inputs1 = std::shared_ptr<Matrix<double>>(new Matrix<double>(2, 1));
    auto inputs2 = std::shared_ptr<Matrix<double>>(new Matrix<double>(2, 1));
    auto inputs3 = std::shared_ptr<Matrix<double>>(new Matrix<double>(2, 1));
    auto inputs4 = std::shared_ptr<Matrix<double>>(new Matrix<double>(2, 1));

    (*inputs1)[0][0] = inputs[0];
    (*inputs1)[1][0] = inputs[4];

    (*inputs2)[0][0] = inputs[1];
    (*inputs2)[1][0] = inputs[5];

    (*inputs3)[0][0] = inputs[2];
    (*inputs3)[1][0] = inputs[6];

    (*inputs4)[0][0] = inputs[3];
    (*inputs4)[1][0] = inputs[7];

    auto outputs1 = std::shared_ptr<Matrix<double>>(new Matrix<double>(2, 1));
    auto outputs2 = std::shared_ptr<Matrix<double>>(new Matrix<double>(2, 1));
    auto outputs3 = std::shared_ptr<Matrix<double>>(new Matrix<double>(2, 1));
    auto outputs4 = std::shared_ptr<Matrix<double>>(new Matrix<double>(2, 1));

    (*outputs1)[0][0] = y[0];
    (*outputs1)[1][0] = y[4];

    (*outputs2)[0][0] = y[1];
    (*outputs2)[1][0] = y[5];

    (*outputs3)[0][0] = y[2];
    (*outputs3)[1][0] = y[6];

    (*outputs4)[0][0] = y[3];
    (*outputs4)[1][0] = y[7];


    double learningRate = 0.01;

    ComputationalGraph *computationalGraphPtr = new ComputationalGraph();
    auto &computationalGraph = *computationalGraphPtr;

    double l2reg = 0.000000;
    // First layer - hidden
    computationalGraph.addDenseLayer(
            {
                    {"width",     INPUT_DIMENSIONS},
                    {"height",    L1_NUM_OF_NEURONS},
                    {"batchSize", BATCH_SIZE}
            },
            normalInitializer,
            ActivationFunction::relu,
            l2reg);

    // Second layer - output
    computationalGraph.addDenseLayer(
            {
                    {"width",     L1_NUM_OF_NEURONS},
                    {"height",    OUTPUT_CLASSES},
                    {"batchSize", BATCH_SIZE}
            },
            normalInitializer2,
            ActivationFunction::softmax);


    std::vector<std::shared_ptr<Matrix<double>>> inputsVec = {
            inputs1,
            inputs2,
            inputs3,
            inputs4
    };
    std::vector<std::shared_ptr<Matrix<double>>> outputsVec = {
            outputs1,
            outputs2,
            outputs3,
            outputs4
    };
    std::vector<int> indices = {0, 1, 2, 3};

    AdamOptimizer optimizer(*computationalGraphPtr, inputsVec, outputsVec, indices, BATCH_SIZE, learningRate);
    for (auto i = 0; i < 10000; i++) {
        optimizer.train();
    }

    auto s = computationalGraph.forwardPass(inputsMatrix);
    s->printValues();

    free(normalInitializer);
    free(normalInitializer2);
}

