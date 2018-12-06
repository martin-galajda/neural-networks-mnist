//
// Created by Martin Galajda on 06/12/2018.
//

#include "./test_utils.h"

void assertSameMatrices(MatrixDoubleSharedPtr expectedMatrix, MatrixDoubleSharedPtr gotMatrix) {
  EXPECT_EQ(expectedMatrix->getNumOfRows(), gotMatrix->getNumOfRows());
  EXPECT_EQ(expectedMatrix->getNumOfCols(), gotMatrix->getNumOfCols());
  EXPECT_EQ(expectedMatrix->getDepth(), gotMatrix->getDepth());
  EXPECT_EQ(expectedMatrix->getBatchSize(), gotMatrix->getBatchSize());

  for (auto b = 0; b < expectedMatrix->getBatchSize(); b++) {
    for (auto i = 0; i < expectedMatrix->getNumOfRows(); i++) {
      for (auto j = 0; j < expectedMatrix->getNumOfCols(); j++) {
        for (auto d = 0; d < expectedMatrix->getDepth(); d++) {

          auto valueExpected = *(*expectedMatrix)(i,j,d,b);
          auto valueGot = *(*gotMatrix)(i,j,d,b);

          EXPECT_DOUBLE_EQ(valueExpected, valueGot);

          if (fabs(valueExpected - valueGot) >= 0.0001) {
            std::cout << "[i][j][d][b] === " << "[" << i << "],[" << j << "],[" << d << "],[" << b << "]" << std::endl;
          }
        }
      }
    }
  }
}

void assertSameMatrices(MatrixDoubleSharedPtr expectedMatrix, Matrix<double>  &gotMatrixRef) {
  auto gotMatrix = &gotMatrixRef;

  EXPECT_EQ(expectedMatrix->getNumOfRows(), gotMatrix->getNumOfRows());
  EXPECT_EQ(expectedMatrix->getNumOfCols(), gotMatrix->getNumOfCols());
  EXPECT_EQ(expectedMatrix->getDepth(), gotMatrix->getDepth());
  EXPECT_EQ(expectedMatrix->getBatchSize(), gotMatrix->getBatchSize());

  for (auto b = 0; b < expectedMatrix->getBatchSize(); b++) {
    for (auto i = 0; i < expectedMatrix->getNumOfRows(); i++) {
      for (auto j = 0; j < expectedMatrix->getNumOfCols(); j++) {
        for (auto d = 0; d < expectedMatrix->getDepth(); d++) {

          auto valueExpected = *(*expectedMatrix)(i,j,d,b);
          auto valueGot = *(*gotMatrix)(i,j,d,b);

          EXPECT_DOUBLE_EQ(valueExpected, valueGot);

          if (fabs(valueExpected - valueGot) >= 0.0001) {
            std::cout << "[i][j][d][b] === " << "[" << i << "],[" << j << "],[" << d << "],[" << b << "]" << std::endl;
          }
        }
      }
    }
  }
}