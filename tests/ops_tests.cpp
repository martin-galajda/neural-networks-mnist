//
// Created by Martin Galajda on 29/11/2018.
//

#include "gtest/gtest.h"
#include "../ops/Convolution.h"


bool areSame(double a, double b, double EPSILON = 0.0001)
{
  return fabs(a - b) < EPSILON;
}


TEST(opsTests, convolution)
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

  auto output = new MatrixDouble(3, 3);

  convolution(inputMatrix, output, kernel, 2);

  double expectedOutputData[] =
    {
      0,  2, 1,
      -6,  -3, 4,
      4,  -3, 0
    };

  auto expectedOutPutDataMatrix = MatrixDoubleSharedPtr(new MatrixDouble(expectedOutputData, 3, 3, true));
  auto &outputRef = *output;

  for (auto i = 0; i < expectedOutPutDataMatrix->getNumOfRows(); i++) {
    for (auto j = 0; j < expectedOutPutDataMatrix->getNumOfCols(); j++) {
      EXPECT_DOUBLE_EQ(outputRef[i][j], (*expectedOutPutDataMatrix)[i][j]);

      if (!areSame(outputRef[i][j], (*expectedOutPutDataMatrix)[i][j])) {
        std::cout << "[i][j] === " << "[" << i << "]" << ", " << "[" << j << "]" << std::endl;
      }
    }
  }

  free(kernel);
  free(output);
}