//
// Created by Martin Galajda on 30/11/2018.
//


#include "gtest/gtest.h"
#include "../ops/Convolution.h"
#include "../layers/ConvLayer.h"
#include "../initializers/ZeroInitializer.h"



TEST(convLayerTest, forwardPropagate)
{
  double input[] =
    {
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 2, 1, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 2, 2, 0,
      0, 1, 0, 2, 0, 1, 0,
      0, 2, 1, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0,


      0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 0, 0, 0,
      0, 2, 1, 2, 0, 1, 0,
      0, 0, 2, 1, 1, 2, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 2, 2, 0,
      0, 0, 0, 0, 0, 0, 0,


      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 1, 1, 0,
      0, 1, 1, 2, 1, 0, 0,
      0, 0, 1, 1, 1, 2, 0,
      0, 0, 2, 1, 1, 0, 0,
      0, 1, 2, 1, 2, 1, 0,
      0, 0, 0, 0, 0, 0, 0

    };


  double kernelData[] =
    {
      -1,  -1,  -1,
      0,   1,  -1,
      -1,   0,   0,

      0,  -1,  -1,
      -1,   1,  -1,
      0,   0,   1,

      1,  -1,   1,
      0,   0,   1,
      1,  -1,  -1
    };

//  double expectedKernelDerivatives[] =
//    {
//       0.  1.  5. // first row, first col all depths
//       5.  5.  4. // first row, snd col all depth
//       0.  1.  5. // first row, third coll all depth
//
//       6.  7.  9. // snd row, first coll, all depth
//       8. 10.  9.
//       6.  7.  9.
//
//       0.  1.  5.
//       5.  5.  4.
//       0.  1.  5.

//  [0,5,0,  /// THIS IS OUR FORMAT
//    6,8,6,
//    0,5,0,
//
//
//    1,5,1,
//    7,10,7,
//    1,5,1,
//
//
//    5,4,5,
//    9,9,9,
//    5,4,5,
//
//
//  ]


//    };

  auto inputMatrix = MatrixDoubleSharedPtr(new MatrixDouble(input, 7, 7, true, 3));

  auto kernel = new MatrixDouble(kernelData, 3, 3, false, 3);

  auto initializer = new ZeroInitializer();

  auto convLayer = new ConvLayer(kernel->getNumOfCols(), kernel->getNumOfRows(), 1, kernel->getDepth(), 1, initializer, ActivationFunction::linear, 2);


  for (auto i = 0; i < kernel->getNumOfRows(); i++) {
    for (auto j = 0; j < kernel->getNumOfCols(); j++) {
      for (auto k = 0; k < kernel->getDepth(); k++) {
        auto &weights = convLayer->getWeights();

        *weights(i, j, k) = *(*kernel)(i, j, k);
      }
    }
  }

  convLayer->getWeights().printValues();

  double expectedOutputData[] =
    {
      0,  2, 1,
      -6,  -3, 4,
      4,  -3, 0
    };

  auto expectedOutPutDataMatrix = MatrixDoubleSharedPtr(new MatrixDouble(expectedOutputData, 3, 3, true));

  auto output = convLayer->forwardPropagate(inputMatrix);
  auto &outputRef = *output;

  outputRef.printValues();

  for (auto i = 0; i < expectedOutPutDataMatrix->getNumOfRows(); i++) {
    for (auto j = 0; j < expectedOutPutDataMatrix->getNumOfCols(); j++) {
      EXPECT_DOUBLE_EQ(outputRef[i][j], (*expectedOutPutDataMatrix)[i][j]);

      if (fabs(outputRef[i][j] - ((*expectedOutPutDataMatrix)[i][j])) >= 0.0001) {
        std::cout << "[i][j] === " << "[" << i << "]" << ", " << "[" << j << "]" << std::endl;
      }
    }
  }

  free(kernel);
  free(convLayer);
}

TEST(convLayerTest, backPropagate)
{
  double input[] =
    {
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 2, 1, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 2, 2, 0,
      0, 1, 0, 2, 0, 1, 0,
      0, 2, 1, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0,


      0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 0, 0, 0,
      0, 2, 1, 2, 0, 1, 0,
      0, 0, 2, 1, 1, 2, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 2, 2, 0,
      0, 0, 0, 0, 0, 0, 0,


      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 1, 1, 0,
      0, 1, 1, 2, 1, 0, 0,
      0, 0, 1, 1, 1, 2, 0,
      0, 0, 2, 1, 1, 0, 0,
      0, 1, 2, 1, 2, 1, 0,
      0, 0, 0, 0, 0, 0, 0

    };


  double kernelData[] =
    {
      -1,  -1,  -1,
      0,   1,  -1,
      -1,   0,   0,

      0,  -1,  -1,
      -1,   1,  -1,
      0,   0,   1,

      1,  -1,   1,
      0,   0,   1,
      1,  -1,  -1
    };


  auto inputMatrix = MatrixDoubleSharedPtr(new MatrixDouble(input, 7, 7, true, 3));

  auto kernel = new MatrixDouble(kernelData, 3, 3, false, 3);

  auto initializer = new ZeroInitializer();

  auto convLayer = new ConvLayer(kernel->getNumOfCols(), kernel->getNumOfRows(), 1, kernel->getDepth(), 1, initializer, ActivationFunction::linear, 2);


  for (auto i = 0; i < kernel->getNumOfRows(); i++) {
    for (auto j = 0; j < kernel->getNumOfCols(); j++) {
      for (auto k = 0; k < kernel->getDepth(); k++) {
        auto &weights = convLayer->getWeights();

        *weights(i, j, k) = *(*kernel)(i, j, k);
      }
    }
  }


  double expectedDerivativesKernel[] =
    {
//    0,5,0,
//    6,8,6,
//    0,5,0,
//
//
//    1,5,1,
//    7,10,7,
//    1,5,1,
//
//
//    5,4,5,
//    9,9,9,
//    5,4,5

      0,26,0,
      31,30,19,
      0,17,0,


      4,18,2,
      44,45,26,
      2,9,1,


      31,16,18,
      47,39,28,
      19,9,11,

    };

  double expectedDerivativesInputs[] =
    {
//      -1, -1, -2, -1, -2, -1, -1,
//      0, 1, -1, 1, -1, 1, -1,
//      -2, -1, -3, -1, -3, -1, -1,
//      0, 1, -1, 1, -1, 1, -1,
//      -2, -1, -3, -1, -3, -1, -1,
//      0, 1, -1, 1, -1, 1, -1,
//      -1, 0, -1, 0, -1, 0, 0,
//
//
//      0, -1, -1, -1, -1, -1, -1,
//      -1, 1, -2, 1, -2, 1, -1,
//      0, -1, 0, -1, 0, -1, 0,
//      -1, 1, -2, 1, -2, 1, -1,
//      0, -1, 0, -1, 0, -1, 0,
//      -1, 1, -2, 1, -2, 1, -1,
//      0, 0, 1, 0, 1, 0, 1,
//
//
//      1, -1, 2, -1, 2, -1, 1,
//      0, 0, 1, 0, 1, 0, 1,
//      2, -2, 2, -2, 2, -2, 0,
//      0, 0, 1, 0, 1, 0, 1,
//      2, -2, 2, -2, 2, -2, 0,
//      0, 0, 1, 0, 1, 0, 1,
//      1, -1, 0, -1, 0, -1, -1

      -1, -1, -3, -2, -5, -3, -3,
      0, 1, -1, 2, -2, 3, -3,
      -3, -2, -8, -4, -13, -6, -6,
      0, 2, -2, 4, -4, 6, -6,
      -5, -3, -13, -6, -21, -9, -9,
      0, 3, -3, 6, -6, 9, -9,
      -3, 0, -6, 0, -9, 0, 0,


      0, -1, -1, -2, -2, -3, -3,
      -1, 1, -3, 2, -5, 3, -3,
      0, -2, -1, -4, -2, -6, -3,
      -2, 2, -6, 4, -10, 6, -6,
      0, -3, -1, -6, -2, -9, -3,
      -3, 3, -9, 6, -15, 9, -9,
      0, 0, 3, 0, 6, 0, 9,


      1, -1, 3, -2, 5, -3, 3,
      0, 0, 1, 0, 2, 0, 3,
      3, -3, 7, -6, 11, -9, 3,
      0, 0, 2, 0, 4, 0, 6,
      5, -5, 11, -10, 17, -15, 3,
      0, 0, 3, 0, 6, 0, 9,
      3, -3, 3, -6, 3, -9, -9
    };

  double forwardDerivativesData[] = {
    1, 2, 3,
    2, 4, 6,
    3, 6, 9
  };

  auto forwaredDerivativesMatrix = MatrixDoubleSharedPtr(new MatrixDouble(forwardDerivativesData, 3, 3, true));
  auto expectedDerivativesMatrix = MatrixDoubleSharedPtr(new MatrixDouble(expectedDerivativesKernel, 3, 3, true, 3));
  auto expectedDerivativesInputsMatrix = MatrixDoubleSharedPtr(new MatrixDouble(expectedDerivativesInputs, 7, 7, true, 3));

  convLayer->forwardPropagate(inputMatrix);
  auto inputDerivativesPtr = convLayer->backPropagate(forwaredDerivativesMatrix);
  auto &inputDerivatives = *inputDerivativesPtr;

  auto &kernelDerivatives = convLayer->getWeightsDerivatives();

  kernelDerivatives.printValues();

  for (auto i = 0; i < kernelDerivatives.getNumOfRows(); i++) {
    for (auto j = 0; j < kernelDerivatives.getNumOfCols(); j++) {
      for (auto z = 0; z < kernelDerivatives.getDepth(); z++) {

        auto valueExpected = *(*expectedDerivativesMatrix)(i, j, z);
        auto valueGot = *kernelDerivatives(i, j, z);

        EXPECT_DOUBLE_EQ(valueExpected, valueGot);

        if (fabs(valueExpected - valueGot) >= 0.0001) {
          std::cout << "[i][j][z] === " << "[" << i << "]" << ", " << "[" << j << "]," << "[" << z << "]" << std::endl;
        }
      }
    }
  }


  for (auto i = 0; i < inputDerivatives.getNumOfRows(); i++) {
    for (auto j = 0; j < inputDerivatives.getNumOfCols(); j++) {
      for (auto z = 0; z < inputDerivatives.getDepth(); z++) {

        auto valueExpected = *(*expectedDerivativesInputsMatrix)(i, j, z);
        auto valueGot = *inputDerivatives(i, j, z);

        EXPECT_DOUBLE_EQ(valueExpected, valueGot);

        if (fabs(valueExpected - valueGot) >= 0.0001) {
          std::cout << "[i][j][z] === " << "[" << i << "]" << ", " << "[" << j << "]," << "[" << z << "]" << std::endl;
        }
      }
    }
  }

  free(kernel);
  free(convLayer);
  free(initializer);
}

TEST(convLayerTest, backPropWithMultipleFilters) {
  double input[] =
    {
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 2, 1, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 2, 2, 0,
      0, 1, 0, 2, 0, 1, 0,
      0, 2, 1, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0,


      0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 0, 0, 0,
      0, 2, 1, 2, 0, 1, 0,
      0, 0, 2, 1, 1, 2, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 2, 2, 0,
      0, 0, 0, 0, 0, 0, 0,


      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 1, 1, 0,
      0, 1, 1, 2, 1, 0, 0,
      0, 0, 1, 1, 1, 2, 0,
      0, 0, 2, 1, 1, 0, 0,
      0, 1, 2, 1, 2, 1, 0,
      0, 0, 0, 0, 0, 0, 0

    };


  double kernelData[] =
    {
     // filter 1
      -1,  -1,  -1,
      0,   1,  -1,
      -1,   0,   0,

      0,  -1,  -1,
      -1,   1,  -1,
      0,   0,   1,

      1,  -1,   1,
      0,   0,   1,
      1,  -1,  -1,

      // filter 2
      1,  -1,  -1,
      2,   1,  -1,
      3,   0,   0,

      0,   5,  -1,
      -1,  3,  -1,
      0,   0,   2,

      1,   3,   1,
      0,   0,   2,
      1,  -1,  -1

    };


  auto inputMatrix = MatrixDoubleSharedPtr(new MatrixDouble(input, 7, 7, true, 3));

  auto kernel = new MatrixDouble(kernelData, 3, 3, false, 3, 2);

  auto initializer = new ZeroInitializer();

  auto convLayer = new ConvLayer(kernel->getNumOfCols(), kernel->getNumOfRows(), 1, kernel->getDepth(), kernel->getBatchSize(), initializer, ActivationFunction::linear, 2);


  for (auto z = 0; z < kernel->getBatchSize(); z++) {
    for (auto i = 0; i < kernel->getNumOfRows(); i++) {
      for (auto j = 0; j < kernel->getNumOfCols(); j++) {
        for (auto k = 0; k < kernel->getDepth(); k++) {
          auto &weights = convLayer->getWeights();

          *weights(i, j, k, z) = *(*kernel)(i, j, k, z);
        }
      }
    }
  }


  double expectedDerivativesKernel[] =
    {

      // FILTER 1
      0,26,0,
      31,30,19,
      0,17,0,


      4,18,2,
      44,45,26,
      2,9,1,


      31,16,18,
      47,39,28,
      19,9,11,


      // FILTER 2
      0,27,0,
      35,33,21,
      0,19,0,


      4,20,3,
      44,50,28,
      4,15,2,


      31,17,19,
      51,41,33,
      21,14,16
    };

  double expectedDerivativesInputs[] =
    {
      1, -3, -1, -6, -6, -6, -6,
      4, 3, 5, 6, 0, 6, -6,
      6, -5, 5, -8, -2, -12, -12,
      6, 5, 3, 8, 4, 12, -12,
      7, -6, 2, -12, 0, -18, -18,
      6, 6, 6, 12, 6, 18, -18,
      6, 0, 12, 0, 18, 0, 0,


      0, 9, -3, 18, -6, 12, -6,
      -3, 7, -9, 14, -12, 12, -6,
      0, 13, 0, 16, 2, 24, -3,
      -5, 11, -13, 16, -20, 24, -12,
      0, 12, 2, 24, 0, 36, 0,
      -6, 12, -18, 24, -30, 36, -18,
      0, 0, 9, 0, 18, 0, 27,


      3, 5, 9, 10, 12, 6, 6,
      0, 0, 5, 0, 10, 0, 9,
      8, 4, 16, 2, 20, 6, 6,
      0, 0, 8, 0, 12, 0, 18,
      11, 1, 21, 4, 34, 6, 6,
      0, 0, 9, 0, 18, 0, 27,
      6, -6, 6, -12, 6, -18, -18
    };

  double forwardDerivativesData[] = {
    1, 2, 3,
    2, 4, 6,
    3, 6, 9,

    2, 4, 3,
    3, 4, 6,
    3, 6, 9

  };

  auto forwaredDerivativesMatrix = MatrixDoubleSharedPtr(new MatrixDouble(forwardDerivativesData, 3, 3, true, 2));
  auto expectedDerivativesMatrix = MatrixDoubleSharedPtr(new MatrixDouble(expectedDerivativesKernel, 3, 3, true, 3, 2));
  auto expectedDerivativesInputsMatrix = MatrixDoubleSharedPtr(new MatrixDouble(expectedDerivativesInputs, 7, 7, true, 3));

  convLayer->forwardPropagate(inputMatrix);
  auto inputDerivativesPtr = convLayer->backPropagate(forwaredDerivativesMatrix);
  auto &inputDerivatives = *inputDerivativesPtr;

  auto &kernelDerivatives = convLayer->getWeightsDerivatives();

  kernelDerivatives.printValues();

  for (auto b = 0; b < kernelDerivatives.getBatchSize(); b++) {
    for (auto i = 0; i < kernelDerivatives.getNumOfRows(); i++) {
      for (auto j = 0; j < kernelDerivatives.getNumOfCols(); j++) {
        for (auto d = 0; d < kernelDerivatives.getDepth(); d++) {

          auto valueExpected = *(*expectedDerivativesMatrix)(i, j, d, b);
          auto valueGot = *kernelDerivatives(i, j, d, b);

          EXPECT_DOUBLE_EQ(valueExpected, valueGot);

          if (fabs(valueExpected - valueGot) >= 0.0001) {
            std::cout << "[i][j][d][b] === " << "[" << i << "],[" << j << "],[" << d << "],[" << b << "]" << std::endl;
          }
        }
      }
    }
  }



  for (auto i = 0; i < inputDerivatives.getNumOfRows(); i++) {
    for (auto j = 0; j < inputDerivatives.getNumOfCols(); j++) {
      for (auto z = 0; z < inputDerivatives.getDepth(); z++) {

        auto valueExpected = *(*expectedDerivativesInputsMatrix)(i, j, z);
        auto valueGot = *inputDerivatives(i, j, z);

        EXPECT_DOUBLE_EQ(valueExpected, valueGot);

        if (fabs(valueExpected - valueGot) >= 0.0001) {
          std::cout << "[i][j][z] === " << "[" << i << "]" << ", " << "[" << j << "]," << "[" << z << "]" << std::endl;
        }
      }
    }
  }

  free(kernel);
  free(convLayer);
  free(initializer);
}


TEST(convLayerTest, backPropagateMultipleFiltersBatched)
{
  double input[] =
    {

      // input sample no 1

      // first dimension
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 2, 1, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 2, 2, 0,
      0, 1, 0, 2, 0, 1, 0,
      0, 2, 1, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0,

      // snd dimension
      0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 0, 0, 0,
      0, 2, 1, 2, 0, 1, 0,
      0, 0, 2, 1, 1, 2, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 2, 2, 0,
      0, 0, 0, 0, 0, 0, 0,

      // third dimension
      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 1, 1, 0,
      0, 1, 1, 2, 1, 0, 0,
      0, 0, 1, 1, 1, 2, 0,
      0, 0, 2, 1, 1, 0, 0,
      0, 1, 2, 1, 2, 1, 0,
      0, 0, 0, 0, 0, 0, 0,

      // input sample no 2


      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 1, 1, 0,
      0, 1, 1, 2, 1, 0, 0,
      0, 0, 1, 1, 1, 2, 0,
      0, 0, 2, 1, 1, 0, 0,
      0, 1, 2, 1, 2, 1, 0,
      0, 0, 0, 0, 0, 0, 0,

      0, 0, 0, 0, 0, 0, 0,
      0, 0, 2, 1, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 2, 2, 0,
      0, 1, 0, 2, 0, 1, 0,
      0, 2, 1, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0,

      0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 0, 0, 0,
      0, 2, 1, 2, 0, 1, 0,
      0, 0, 2, 1, 1, 2, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 2, 2, 0,
      0, 0, 0, 0, 0, 0, 0
    };


  double kernelData[] =
    {
      // filter 1
      -1,  -1,  -1,
      0,   1,  -1,
      -1,   0,   0,

      0,  -1,  -1,
      -1,   1,  -1,
      0,   0,   1,

      1,  -1,   1,
      0,   0,   1,
      1,  -1,  -1,

      // filter 2
      1,  -1,  -1,
      2,   1,  -1,
      3,   0,   0,

      0,   5,  -1,
      -1,  3,  -1,
      0,   0,   2,

      1,   3,   1,
      0,   0,   2,
      1,  -1,  -1

    };


  auto inputMatrix = MatrixDoubleSharedPtr(new MatrixDouble(input, 7, 7, true, 3, 2));

  auto kernel = new MatrixDouble(kernelData, 3, 3, false, 3, 2);

  auto initializer = new ZeroInitializer();

  auto convLayer = new ConvLayer(kernel->getNumOfCols(), kernel->getNumOfRows(), inputMatrix->getBatchSize(), kernel->getDepth(), kernel->getBatchSize(), initializer, ActivationFunction::linear, 2);

  for (auto z = 0; z < kernel->getBatchSize(); z++) {
    for (auto i = 0; i < kernel->getNumOfRows(); i++) {
      for (auto j = 0; j < kernel->getNumOfCols(); j++) {
        for (auto k = 0; k < kernel->getDepth(); k++) {
          auto &weights = convLayer->getWeights();

          *weights(i, j, k, z) = *(*kernel)(i, j, k, z);
        }
      }
    }
  }


  double expectedDerivativesKernel[] =
    {
      // FILTER 1
      31,42,18,
      78,69,47,
      19,26,11,

      4,44,2,
      75,75,45,
      2,26,1,

      35,34,20,
      91,84,54,
      21,18,12,

      // FILTER 2
      31,44,19,
      86,74,54,
      21,33,16,

      4,47,3,
      79,83,49,
      4,34,2,

      35,37,22,
      95,91,61,
      25,29,18,
    };

  double expectedDerivativesInputs[] =
    {
      // BATCH 1
      1, -3, -1, -6, -6, -6, -6,
      4, 3, 5, 6, 0, 6, -6,
      6, -5, 5, -8, -2, -12, -12,
      6, 5, 3, 8, 4, 12, -12,
      7, -6, 2, -12, 0, -18, -18,
      6, 6, 6, 12, 6, 18, -18,
      6, 0, 12, 0, 18, 0, 0,


      0, 9, -3, 18, -6, 12, -6,
      -3, 7, -9, 14, -12, 12, -6,
      0, 13, 0, 16, 2, 24, -3,
      -5, 11, -13, 16, -20, 24, -12,
      0, 12, 2, 24, 0, 36, 0,
      -6, 12, -18, 24, -30, 36, -18,
      0, 0, 9, 0, 18, 0, 27,


      3, 5, 9, 10, 12, 6, 6,
      0, 0, 5, 0, 10, 0, 9,
      8, 4, 16, 2, 20, 6, 6,
      0, 0, 8, 0, 12, 0, 18,
      11, 1, 21, 4, 34, 6, 6,
      0, 0, 9, 0, 18, 0, 27,
      6, -6, 6, -12, 6, -18, -18,


      // BATCH 2
      1, -3, -1, -6, -6, -6, -6,
      4, 3, 5, 6, 0, 6, -6,
      6, -5, 5, -8, -2, -12, -12,
      6, 5, 3, 8, 4, 12, -12,
      7, -6, 2, -12, 0, -18, -18,
      6, 6, 6, 12, 6, 18, -18,
      6, 0, 12, 0, 18, 0, 0,


      0, 9, -3, 18, -6, 12, -6,
      -3, 7, -9, 14, -12, 12, -6,
      0, 13, 0, 16, 2, 24, -3,
      -5, 11, -13, 16, -20, 24, -12,
      0, 12, 2, 24, 0, 36, 0,
      -6, 12, -18, 24, -30, 36, -18,
      0, 0, 9, 0, 18, 0, 27,


      3, 5, 9, 10, 12, 6, 6,
      0, 0, 5, 0, 10, 0, 9,
      8, 4, 16, 2, 20, 6, 6,
      0, 0, 8, 0, 12, 0, 18,
      11, 1, 21, 4, 34, 6, 6,
      0, 0, 9, 0, 18, 0, 27,
      6, -6, 6, -12, 6, -18, -18
    };

  double forwardDerivativesData[] = {
    1, 2, 3,
    2, 4, 6,
    3, 6, 9,

    2, 4, 3,
    3, 4, 6,
    3, 6, 9,

    1, 2, 3,
    2, 4, 6,
    3, 6, 9,

    2, 4, 3,
    3, 4, 6,
    3, 6, 9

  };

  auto forwaredDerivativesMatrix = MatrixDoubleSharedPtr(new MatrixDouble(forwardDerivativesData, 3, 3, true, 2, 2));
  auto expectedDerivativesMatrix = MatrixDoubleSharedPtr(new MatrixDouble(expectedDerivativesKernel, 3, 3, true, 3, 2));
  auto expectedDerivativesInputsMatrix = MatrixDoubleSharedPtr(new MatrixDouble(expectedDerivativesInputs, 7, 7, true, 3, 2));

  convLayer->forwardPropagate(inputMatrix);
  auto inputDerivativesPtr = convLayer->backPropagate(forwaredDerivativesMatrix);
  auto &inputDerivatives = *inputDerivativesPtr;

  auto &kernelDerivatives = convLayer->getWeightsDerivatives();

  kernelDerivatives.printValues();

  for (auto b = 0; b < kernelDerivatives.getBatchSize(); b++) {
    for (auto i = 0; i < kernelDerivatives.getNumOfRows(); i++) {
      for (auto j = 0; j < kernelDerivatives.getNumOfCols(); j++) {
        for (auto d = 0; d < kernelDerivatives.getDepth(); d++) {

          auto valueExpected = *(*expectedDerivativesMatrix)(i, j, d, b);
          auto valueGot = *kernelDerivatives(i, j, d, b);

          EXPECT_DOUBLE_EQ(valueExpected, valueGot);

          if (fabs(valueExpected - valueGot) >= 0.0001) {
            std::cout << "[i][j][d][b] === " << "[" << i << "],[" << j << "],[" << d << "],[" << b << "]" << std::endl;
          }
        }
      }
    }
  }

  inputDerivatives.printValues();

  for (auto b = 0; b < inputDerivatives.getBatchSize(); b++) {
    for (auto i = 0; i < inputDerivatives.getNumOfRows(); i++) {
      for (auto j = 0; j < inputDerivatives.getNumOfCols(); j++) {
        for (auto z = 0; z < inputDerivatives.getDepth(); z++) {

          auto valueExpected = *(*expectedDerivativesInputsMatrix)(i, j, z, b);
          auto valueGot = *inputDerivatives(i, j, z, b);

          EXPECT_DOUBLE_EQ(valueExpected, valueGot);

          if (fabs(valueExpected - valueGot) >= 0.0001) {
            std::cout << "[i][j][d][b] === " << "[" << i << "],[" << j << "],[" << z << "],[" << b << "]" << std::endl;
          }
        }
      }
    }
  }


  free(kernel);
  free(convLayer);
  free(initializer);
}



TEST(convLayerTest, backPropagateWithRelu)
{
  double input[] =
    {

      // input sample no 1

      // first dimension
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 2, 1, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 2, 2, 0,
      0, 1, 0, 2, 0, 1, 0,
      0, 2, 1, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0,

      // snd dimension
      0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 0, 0, 0,
      0, 2, 1, 2, 0, 1, 0,
      0, 0, 2, 1, 1, 2, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 2, 2, 0,
      0, 0, 0, 0, 0, 0, 0,

      // third dimension
      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 1, 1, 0,
      0, 1, 1, 2, 1, 0, 0,
      0, 0, 1, 1, 1, 2, 0,
      0, 0, 2, 1, 1, 0, 0,
      0, 1, 2, 1, 2, 1, 0,
      0, 0, 0, 0, 0, 0, 0,

      // input sample no 2


      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 1, 1, 0,
      0, 1, 1, 2, 1, 0, 0,
      0, 0, 1, 1, 1, 2, 0,
      0, 0, 2, 1, 1, 0, 0,
      0, 1, 2, 1, 2, 1, 0,
      0, 0, 0, 0, 0, 0, 0,

      0, 0, 0, 0, 0, 0, 0,
      0, 0, 2, 1, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 2, 2, 0,
      0, 1, 0, 2, 0, 1, 0,
      0, 2, 1, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0,

      0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 0, 0, 0,
      0, 2, 1, 2, 0, 1, 0,
      0, 0, 2, 1, 1, 2, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 2, 2, 0, 2, 2, 0,
      0, 0, 0, 0, 0, 0, 0
    };


  double kernelData[] =
    {
      // filter 1
      -1,  -1,  -1,
      0,   1,  -1,
      -1,   0,   0,

      0,  -1,  -1,
      -1,   1,  -1,
      0,   0,   1,

      1,  -1,   1,
      0,   0,   1,
      1,  -1,  -1,

      // filter 2
      1,  -1,  -1,
      2,   1,  -1,
      3,   0,   0,

      0,   5,  -1,
      -1,  3,  -1,
      0,   0,   2,

      1,   3,   1,
      0,   0,   2,
      1,  -1,  -1

    };


  auto inputMatrix = MatrixDoubleSharedPtr(new MatrixDouble(input, 7, 7, true, 3, 2));

  auto kernel = new MatrixDouble(kernelData, 3, 3, false, 3, 2);

  auto initializer = new ZeroInitializer();

  auto convLayer = new ConvLayer(kernel->getNumOfCols(), kernel->getNumOfRows(), inputMatrix->getBatchSize(), kernel->getDepth(), kernel->getBatchSize(), initializer, ActivationFunction::relu, 2);

  for (auto z = 0; z < kernel->getBatchSize(); z++) {
    for (auto i = 0; i < kernel->getNumOfRows(); i++) {
      for (auto j = 0; j < kernel->getNumOfCols(); j++) {
        for (auto k = 0; k < kernel->getDepth(); k++) {
          auto &weights = convLayer->getWeights();

          *weights(i, j, k, z) = *(*kernel)(i, j, k, z);
        }
      }
    }
  }


  double expectedDerivativesKernel[] =
    {
      0,3,0,
      16,20,3,
      0,6,0,


      0,6,0,
      6,22,6,
      2,7,0,


      6,0,6,
      13,18,8,
      11,4,2,

      31,44,19,
      86,70,50,
      21,31,14,


      4,47,3,
      79,83,45,
      4,32,2,


      35,37,22,
      95,89,61,
      25,25,16,

    };

  double expectedDerivativesInputs[] =
    {
      2, -2, 0, -6, -6, -6, -6,
      4, 2, 6, 6, 0, 6, -6,
      9, -3, 11, -4, 2, -12, -12,
      6, 3, 5, 4, 8, 12, -12,
      9, -6, 12, -6, 15, -9, -9,
      6, 6, 6, 6, 12, 9, -9,
      6, 0, 18, 0, 27, 0, 0,

      0, 10, -2, 18, -6, 12, -6,
      -2, 6, -8, 14, -12, 12, -6,
      0, 15, 1, 20, 6, 24, -3,
      -3, 9, -7, 12, -16, 24, -12,
      0, 12, 0, 30, 2, 45, 9,
      -6, 12, -12, 18, -15, 27, -9,
      0, 0, 9, 0, 12, 0, 18,

      2, 6, 8, 10, 12, 6, 6,
      0, 0, 4, 0, 10, 0, 9,
      5, 7, 11, 6, 16, 6, 6,
      0, 0, 6, 0, 8, 0, 18,
      9, 3, 13, 14, 23, 15, -3,
      0, 0, 9, 0, 12, 0, 18,
      6, -6, 0, -6, 3, -9, -9,




      0, 0, 4, -4, -1, -3, -3,
      0, 0, 8, 4, 2, 3, -3,
      3, -3, 13, -4, 11, -6, -6,
      6, 3, 5, 4, 8, 6, -6,
      12, -3, 15, -6, 21, -9, -9,
      6, 3, 9, 6, 12, 9, -9,
      9, 0, 18, 0, 27, 0, 0,

      0, 0, 0, 20, -4, 15, -3,
      0, 0, -4, 12, -7, 9, -3,
      0, 15, -3, 20, 4, 30, 0,
      -3, 9, -7, 12, -10, 18, -6,
      0, 15, 3, 30, 2, 45, 3,
      -3, 9, -9, 18, -15, 27, -9,
      0, 0, 6, 0, 12, 0, 18,

      0, 0, 4, 12, 7, 9, 3,
      0, 0, 0, 0, 8, 0, 6,
      3, 9, 11, 8, 9, 15, 3,
      0, 0, 6, 0, 8, 0, 12,
      6, 6, 10, 14, 17, 21, 3,
      0, 0, 6, 0, 12, 0, 18,
      3, -3, 3, -6, 3, -9, -9
    };

  double forwardDerivativesData[] = {
    1, 2, 3,
    2, 4, 6,
    3, 6, 9,

    2, 4, 3,
    3, 4, 6,
    3, 6, 9,

    1, 2, 3,
    2, 4, 6,
    3, 6, 9,

    2, 4, 3,
    3, 4, 6,
    3, 6, 9

  };

  double expectedForwardPropagated[] = {
    0, 2, 1,
    0, 0, 4,
    4, 0, 0,


    5, 11, 1,
    11, 20, 18,
    10, 5, 6,

    0, 0, 0,
    0, 0, 0,
    0, 0, 0,

    0, 6, 5,
    16, 19, 15,
    11, 15, 10,
  };

  auto forwaredDerivativesMatrix = MatrixDoubleSharedPtr(new MatrixDouble(forwardDerivativesData, 3, 3, true, 2, 2));
  auto expectedDerivativesMatrix = MatrixDoubleSharedPtr(new MatrixDouble(expectedDerivativesKernel, 3, 3, true, 3, 2));
  auto expectedDerivativesInputsMatrix = MatrixDoubleSharedPtr(new MatrixDouble(expectedDerivativesInputs, 7, 7, true, 3, 2));
  auto expectedForwardPropagatedMatrix = MatrixDoubleSharedPtr(new MatrixDouble(expectedForwardPropagated, 3, 3, true, 2, 2));

  auto forwardPropagated = convLayer->forwardPropagate(inputMatrix);

  forwardPropagated->printValues();

  for (auto b = 0; b < forwardPropagated->getBatchSize(); b++) {
    for (auto i = 0; i < forwardPropagated->getNumOfRows(); i++) {
      for (auto j = 0; j < forwardPropagated->getNumOfCols(); j++) {
        for (auto d = 0; d < forwardPropagated->getDepth(); d++) {

          auto valueExpected = *(*expectedForwardPropagatedMatrix)(i, j, d, b);
          auto valueGot = *(*forwardPropagated)(i, j, d, b);

          EXPECT_DOUBLE_EQ(valueExpected, valueGot);

          if (fabs(valueExpected - valueGot) >= 0.0001) {
            std::cout << "[i][j][d][b] === " << "[" << i << "],[" << j << "],[" << d << "],[" << b << "]" << std::endl;
          }
        }
      }
    }
  }

  auto inputDerivativesPtr = convLayer->backPropagate(forwaredDerivativesMatrix);
  auto &inputDerivatives = *inputDerivativesPtr;

  auto &kernelDerivatives = convLayer->getWeightsDerivatives();

  kernelDerivatives.printValues();

  for (auto b = 0; b < kernelDerivatives.getBatchSize(); b++) {
    for (auto i = 0; i < kernelDerivatives.getNumOfRows(); i++) {
      for (auto j = 0; j < kernelDerivatives.getNumOfCols(); j++) {
        for (auto d = 0; d < kernelDerivatives.getDepth(); d++) {

          auto valueExpected = *(*expectedDerivativesMatrix)(i, j, d, b);
          auto valueGot = *kernelDerivatives(i, j, d, b);

          EXPECT_DOUBLE_EQ(valueExpected, valueGot);

          if (fabs(valueExpected - valueGot) >= 0.0001) {
            std::cout << "[i][j][d][b] === " << "[" << i << "],[" << j << "],[" << d << "],[" << b << "]" << std::endl;
          }
        }
      }
    }
  }

  inputDerivatives.printValues();

  for (auto b = 0; b < inputDerivatives.getBatchSize(); b++) {
    for (auto i = 0; i < inputDerivatives.getNumOfRows(); i++) {
      for (auto j = 0; j < inputDerivatives.getNumOfCols(); j++) {
        for (auto z = 0; z < inputDerivatives.getDepth(); z++) {

          auto valueExpected = *(*expectedDerivativesInputsMatrix)(i, j, z, b);
          auto valueGot = *inputDerivatives(i, j, z, b);

          EXPECT_DOUBLE_EQ(valueExpected, valueGot);

          if (fabs(valueExpected - valueGot) >= 0.0001) {
            std::cout << "[i][j][d][b] === " << "[" << i << "],[" << j << "],[" << z << "],[" << b << "]" << std::endl;
          }
        }
      }
    }
  }


  free(kernel);
  free(convLayer);
  free(initializer);
}