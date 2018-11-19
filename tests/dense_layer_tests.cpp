//
// Created by Martin Galajda on 27/10/2018.
//

#include "gtest/gtest.h"
#include "../matrix_impl/Matrix.hpp"
#include "../layers/DenseLayer.h"
#include "Matrix.cpp"
#include <cmath>


TEST(denseLayerTests, forwardPropagationWithBatchSizeOne)
{
    double initialWeightsL1[] = {
            1.0, 1.0,
            1.0, 1.0,
            1.0, 1.0,
    };
    auto *l1 = new DenseLayer(
            2 /* width */,
            3 /* height */,
            1,
            initialWeightsL1,
            ActivationFunction::relu
    );

    double dataForA[] = { 0.0, 1.0 };
    auto input = std::shared_ptr<Matrix<double>>(new Matrix<double>(dataForA, 2, 1));

    auto z1 = l1->forwardPropagate(input);
    double expectedValues[] = { 1.0, 1.0, 1.0 };

    EXPECT_DOUBLE_EQ((*z1)[0][0], expectedValues[0]);
    EXPECT_DOUBLE_EQ((*z1)[1][0], expectedValues[1]);
    EXPECT_DOUBLE_EQ((*z1)[2][0], expectedValues[2]);

    z1->printValues();

    double initialWeightsL2[] = {
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
    };
    auto *l2 = new DenseLayer(
            3 /* width */,
            2 /* height */,
            1 /* batch size */,
            initialWeightsL2,
            ActivationFunction::softmax
    );

    auto s = l2->forwardPropagate(z1);
    s->printValues();

    double expectedValuesOutput[] = { 1.0/2, 1.0/2 };
    EXPECT_DOUBLE_EQ((*s)[0][0], expectedValuesOutput[0]);
    EXPECT_DOUBLE_EQ((*s)[1][0], expectedValuesOutput[1]);
}

TEST(denseLayerTests, forwardPropagationWithBatchSizeTwo)
{
    double initialWeightsL1[] = {
            1.0, 1.0,
            1.0, 1.0,
            1.0, 1.0,
    };
    auto *l1 = new DenseLayer(
            2 /* width */,
            3 /* height */,
            2 /* batch size */,
            initialWeightsL1,
            ActivationFunction::relu
    );

    double dataForA[] = {
            1.0, 1.0,
            0.0, 0.0
    };
    auto input = std::shared_ptr<Matrix<double>>(new Matrix<double>(dataForA, 2, 2));

    auto z1 = l1->forwardPropagate(input);
    double expectedValues[] = { 1.0, 1.0, 1.0 };

    EXPECT_DOUBLE_EQ((*z1)[0][0], expectedValues[0]);
    EXPECT_DOUBLE_EQ((*z1)[1][0], expectedValues[1]);
    EXPECT_DOUBLE_EQ((*z1)[2][0], expectedValues[2]);

    EXPECT_DOUBLE_EQ((*z1)[0][1], expectedValues[0]);
    EXPECT_DOUBLE_EQ((*z1)[1][1], expectedValues[1]);
    EXPECT_DOUBLE_EQ((*z1)[2][1], expectedValues[2]);

    z1->printValues();

    double initialWeightsL2[] = {
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
    };
    auto *l2 = new DenseLayer(
            3 /* width */,
            2 /* height */,
            2 /* batch size */,
            initialWeightsL2,
            ActivationFunction::softmax
    );

    auto s = l2->forwardPropagate(z1);
    s->printValues();

    double expectedValuesOutput[] = { 1.0/2, 1.0/2 };
    EXPECT_DOUBLE_EQ((*s)[0][0], expectedValuesOutput[0]);
    EXPECT_DOUBLE_EQ((*s)[1][0], expectedValuesOutput[1]);
    EXPECT_DOUBLE_EQ((*s)[0][1], expectedValuesOutput[0]);
    EXPECT_DOUBLE_EQ((*s)[1][1], expectedValuesOutput[1]);
}

TEST(denseLayerTests, backwardPropagate)
{
    double initialWeightsL2[] = {
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
    };
    auto *l2 = new DenseLayer(
            3 /* width */,
            2 /* height */,
            1 /* batch size */,
            initialWeightsL2,
            ActivationFunction::softmax
    );

    double z1data[] = { 1.0, 1.0, 1.0 };
    auto z1 = std::shared_ptr<Matrix<double >>(new Matrix<double>(z1data, 3, 1));

    auto s = l2->forwardPropagate(z1);

    double cross_entropy_loss_derivatives_data[] = {
             1.0 / 2,
            -1.0 / 2
    };
    auto cross_entropy_loss_derivatives = std::shared_ptr<Matrix<double >>(new Matrix<double>(cross_entropy_loss_derivatives_data, 2, 1));


    auto inputDerivatives = l2->backPropagate(cross_entropy_loss_derivatives);

    EXPECT_DOUBLE_EQ((*inputDerivatives)[0][0], 0);
    EXPECT_DOUBLE_EQ((*inputDerivatives)[1][0], 0);
    EXPECT_DOUBLE_EQ((*inputDerivatives)[2][0], 0);

    auto &weightDerivatives = l2->getWeightsDerivatives();

    EXPECT_DOUBLE_EQ(weightDerivatives[0][0],  1.0/2);
    EXPECT_DOUBLE_EQ(weightDerivatives[0][1],  1.0/2);
    EXPECT_DOUBLE_EQ(weightDerivatives[0][2],  1.0/2);

    EXPECT_DOUBLE_EQ(weightDerivatives[1][0], -1.0/2);
    EXPECT_DOUBLE_EQ(weightDerivatives[1][1], -1.0/2);
    EXPECT_DOUBLE_EQ(weightDerivatives[1][2], -1.0/2);


    free(l2);
}

TEST(denseLayerTests, backwardPropagateBatched)
{
    double initialWeightsL2[] = {
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
    };
    auto *l2 = new DenseLayer(
            3 /* width */,
            2 /* height */,
            2 /* batch size */,
            initialWeightsL2,
            ActivationFunction::softmax
    );

    double z1data[] = {
            1.0, 1.0,
            1.0, 1.0,
            1.0, 1.0
    };
    auto z1 = std::shared_ptr<Matrix<double >>(new Matrix<double>(z1data, 3, 2));

    auto s = l2->forwardPropagate(z1);

    double cross_entropy_loss_derivatives_data[] = {
             1.0 / 2,  1.0 / 2,
            -1.0 / 2,  1.0 / 2
    };
    auto cross_entropy_loss_derivatives = std::shared_ptr<Matrix<double >>(new Matrix<double>(cross_entropy_loss_derivatives_data, 2, 2));

    auto inputDerivatives = l2->backPropagate(cross_entropy_loss_derivatives);

    EXPECT_DOUBLE_EQ((*inputDerivatives)[0][0], 0);
    EXPECT_DOUBLE_EQ((*inputDerivatives)[1][0], 0);
    EXPECT_DOUBLE_EQ((*inputDerivatives)[2][0], 0);

    EXPECT_DOUBLE_EQ((*inputDerivatives)[0][1], 1.0);
    EXPECT_DOUBLE_EQ((*inputDerivatives)[1][1], 1.0);
    EXPECT_DOUBLE_EQ((*inputDerivatives)[2][1], 1.0);

    auto &weightDerivatives = l2->getWeightsDerivatives();

    EXPECT_DOUBLE_EQ(weightDerivatives[0][0], 1.0);
    EXPECT_DOUBLE_EQ(weightDerivatives[0][1], 1.0);
    EXPECT_DOUBLE_EQ(weightDerivatives[0][2], 1.0);

    EXPECT_DOUBLE_EQ(weightDerivatives[1][0], 0);
    EXPECT_DOUBLE_EQ(weightDerivatives[1][1], 0);
    EXPECT_DOUBLE_EQ(weightDerivatives[1][2], 0);

    free(l2);
}
