//
// Created by Martin Galajda on 28/10/2018.
//

#include "gtest/gtest.h"
#include "../data/MNISTParser.h"


TEST(mnistParserTests, trainingData)
{
    MNISTParser parser("../data/mnist_train_vectors.csv", "../data/mnist_train_labels.csv");

    auto instances = parser.parseToMatrices();

    std::cout << std::endl;
    std::cout << (*instances[0]).getNumOfRows() << ", " << (*instances[0]).getNumOfCols() << std::endl;
    std::cout << (*instances[instances.size() - 1]).getNumOfRows() << ", " << (*instances[instances.size() - 1]).getNumOfCols() << std::endl;

    // check manually handpicked value from first row
    ASSERT_DOUBLE_EQ((*instances[0])[683][0], 16.0);

    auto labels = parser.parseLabelsToOneHotEncodedVectors();

    EXPECT_EQ(instances.size(), labels.size());

    // FIRST LABEL IS 5 -> ONLY 6th ELEMENT SHOULD BE 1
    EXPECT_EQ((*labels[0])[5][0], 1);

    EXPECT_EQ((*labels[0])[9][0], 0);
    EXPECT_EQ((*labels[0])[8][0], 0);
    EXPECT_EQ((*labels[0])[7][0], 0);
    EXPECT_EQ((*labels[0])[6][0], 0);
    EXPECT_EQ((*labels[0])[4][0], 0);
    EXPECT_EQ((*labels[0])[3][0], 0);
    EXPECT_EQ((*labels[0])[2][0], 0);
    EXPECT_EQ((*labels[0])[1][0], 0);
    EXPECT_EQ((*labels[0])[0][0], 0);

    // LAST LABEL IS 8 -> ONLY 9th ELEMENT SHOULD BE 1
    EXPECT_EQ((*labels[labels.size() - 1])[8][0], 1);

    EXPECT_EQ((*labels[labels.size() - 1])[9][0], 0);
    EXPECT_EQ((*labels[labels.size() - 1])[7][0], 0);
    EXPECT_EQ((*labels[labels.size() - 1])[6][0], 0);
    EXPECT_EQ((*labels[labels.size() - 1])[5][0], 0);
    EXPECT_EQ((*labels[labels.size() - 1])[4][0], 0);
    EXPECT_EQ((*labels[labels.size() - 1])[3][0], 0);
    EXPECT_EQ((*labels[labels.size() - 1])[2][0], 0);
    EXPECT_EQ((*labels[labels.size() - 1])[1][0], 0);
    EXPECT_EQ((*labels[labels.size() - 1])[0][0], 0);
}