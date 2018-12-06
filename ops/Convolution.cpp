//
// Created by Martin Galajda on 28/11/2018.
//

#include "Convolution.h"
#include <cmath>
#include <vector>

bool is_integer(float k)
{
    return std::floor(k) == k;
}
//
//MatrixDoubleSharedPtr convolution(MatrixDoubleSharedPtr &input, MatrixDoubleSharedPtr &kernel, int stride) {
//    // assume square matrix for now
//    auto outputWidth = ((input->getNumOfCols() - kernel->getNumOfCols()) / stride) + 1;
//    auto outputHeight = ((input->getNumOfRows() - kernel->getNumOfRows()) / stride) + 1;
//
//    if (!is_integer(outputWidth)) {
//        throw std::invalid_argument("Got invalid stride input + kernel + stride combination for width.");
//    }
//
//    if (!is_integer(outputHeight)) {
//        throw std::invalid_argument("Got invalid stride input + kernel + stride combination for height.");
//    }
//
//    auto outputWidthBatched = static_cast<int>(outputWidth * input->getBatchSize());
//    auto outputHeightBatched = static_cast<int>(outputHeight * input->getBatchSize());
//
//    auto convolvedOutput = MatrixDoubleSharedPtr(new Matrix<double>(outputHeightBatched, outputWidthBatched));
//
//    // TODO: convolve output
//    // convole output...
//
//
//    return convolvedOutput;
//}

MatrixDouble *convolution(MatrixDoubleSharedPtr &input, MatrixDouble *outputPtr, MatrixDouble *kernelPtr, int stride) {
    auto &inputMatrix = *input;
    auto &kernel = *kernelPtr;
    auto &output = *outputPtr;

    // assume square matrix for now
    auto outputWidth = ((input->getNumOfCols() - kernel.getNumOfCols()) / stride) + 1;
    auto outputHeight = ((input->getNumOfRows() - kernel.getNumOfRows()) / stride) + 1;

    if (!is_integer(outputWidth)) {
        outputWidth = (int) outputWidth;
    }

    if (!is_integer(outputHeight)) {
      outputWidth = (int) outputHeight;
    }

    if (outputWidth != output.getNumOfCols()) {
      throw std::invalid_argument("Convolution: something wrong! outputWidth != output->getNumOfCols()");
    }

    if (outputHeight != output.getNumOfRows()) {
      throw std::invalid_argument("Convolution: something wrong! outputHeight != output->getNumOfRows()");
    }


    for (auto channelIdx = 0; channelIdx < input->getDepth(); channelIdx++) {
      auto inputIdxStartRow = 0;

      for (auto i = 0; i < output.getNumOfRows(); i++) {
        //do magic here

        auto inputIdxStartCol = 0;
        for (auto j = 0; j < output.getNumOfCols(); j++) {

          for (auto kernelRow = 0; kernelRow < kernel.getNumOfRows(); kernelRow++) {
            for (auto kernelCol = 0; kernelCol < kernel.getNumOfCols(); kernelCol++) {

              if (inputMatrix.getNumOfRows() <= inputIdxStartRow + kernelRow) {
                continue;
              }

              if (inputMatrix.getNumOfCols() <= inputIdxStartCol + kernelCol) {
                continue;
              }

              // do magic here
              *(output(i,j,0,0)) +=
                *kernel(kernelRow, kernelCol, channelIdx)
                *
                (*inputMatrix(inputIdxStartRow + kernelRow, inputIdxStartCol + kernelCol, channelIdx));

            }
          }

          // move pointer
          inputIdxStartCol += stride;
        }

        // move pointer
        inputIdxStartRow += stride;
      }
    }


    return outputPtr;
}