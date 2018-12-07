//
// Created by Martin Galajda on 06/12/2018.
//

#include "report_accuracy.h"

double reportAccuracy(MatrixDoubleSharedPtr predicted, MatrixDoubleSharedPtr expected) {
  auto predictedValues = predicted->argMaxByRow();
  auto expectedValues = expected->argMaxByRow();

  if (predictedValues->getNumOfRows() != expectedValues->getNumOfRows()) {
    throw std::invalid_argument("Mismatched no of examples when computing accuracy.");
  }

  auto matchedPredictions = 0;
  for (auto row = 0; row < predictedValues->getNumOfRows(); row++) {
    if ((*predictedValues)[row][0] == (*expectedValues)[row][0]) {
      matchedPredictions++;
    }
  }

  auto accuracy = (matchedPredictions / 1.0) / predictedValues->getNumOfRows();

//  std::cout << "Batch accuracy: " << accuracy
//            << std::endl;

  return accuracy;
}
