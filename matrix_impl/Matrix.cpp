//
// Created by Martin Galajda on 25/10/2018.
//

#include <cmath>
#include <cstring>
#include "./Matrix.hpp"
#include <sstream>

/*********** CONSTRUCTORS ********************************************************/

template <typename T>
Matrix<T>::Matrix(int rows, int cols, int depth, int batchSize) {
    this->numOfRows = rows;
    this->numOfCols = cols;
    this->depth = depth;
    this->batchSize = batchSize;

    this->rawData = (T*) malloc(sizeof(T) * rows * cols * depth * batchSize);
}

template <typename T>
Matrix<T>::Matrix(int rows, int cols, BaseInitializer *initializer): Matrix(rows, cols, 1, 1) {
  for (auto sampleIdx = 0; sampleIdx < batchSize; sampleIdx++) {
    for (auto depthIdx = 0; depthIdx < depth; depthIdx++) {
      for (auto row = 0; row < numOfRows; row++) {
        for (auto col = 0; col < numOfCols; col++) {
          *(*this)(row,col,depthIdx,sampleIdx) = initializer->getValue();
        }
      }
    }
  }
}

template <typename T>
Matrix<T>::Matrix(int rows, int cols, int depth, int batchSize, BaseInitializer *initializer): Matrix(rows, cols, depth, batchSize) {
    this->batchSize = batchSize;

    for (auto sampleIdx = 0; sampleIdx < batchSize; sampleIdx++) {
      for (auto depthIdx = 0; depthIdx < depth; depthIdx++) {
        for (auto row = 0; row < numOfRows; row++) {
          for (auto col = 0; col < numOfCols; col++) {
            *(*this)(row,col,depthIdx,sampleIdx) = initializer->getValue();
          }
        }
      }
    }
}


template <typename T>
Matrix<T>::Matrix(T *data, int rows, int cols, bool copyData, int depth, int batchSize): Matrix(rows, cols, depth, batchSize) {
    this->numOfRows = rows;
    this->numOfCols = cols;
    this->depth = depth;
    this->batchSize = batchSize;

    if (copyData) {
        memcpy(this->rawData, data, sizeof(T) * rows * cols * depth * batchSize);
    } else {
        this->rawData = data;
        this->notOwningData = true;
    }
}


template <typename T>
void Matrix<T>::initialize(BaseInitializer *initializer){
  for (auto sampleIdx = 0; sampleIdx < batchSize; sampleIdx++) {
    for (auto depthIdx = 0; depthIdx < depth; depthIdx++) {
      for (auto row = 0; row < numOfRows; row++) {
        for (auto col = 0; col < numOfCols; col++) {
          *(*this)(row, col, depthIdx, sampleIdx) = initializer->getValue();
        }
      }
    }
  }
}


/*********** DESTRUCTORS ********************************************************/

template <typename T>
Matrix<T>::~Matrix() {
    if (!this->notOwningData) {
        free(rawData);
    }
}

/*********** OPERATORS ********************************************************/

template <typename T>
inline T* Matrix<T>::operator[](const int& index) {
    return this->rawData + (index * numOfCols);
}

template <typename T>
inline T* Matrix<T>::operator()(const int& row, const int& col, const int& channelIdx) {
//    if (col >= this->numOfCols) {
//        throw std::invalid_argument("Matrix(row,col,channelIdx) out of range: col >= this->numOfCols.");
//    } else if (row >= this->numOfRows) {
//        throw std::invalid_argument("Matrix(row,col,channelIdx) out of range: row >= this->numOfRows.");
//    } else if (channelIdx >= this->depth) {
//        throw std::invalid_argument("Matrix(row,col,channelIdx) out of range: channelIdx >= this->depth.");
//    }

    auto channelOffset = (this->numOfRows * this->numOfCols * channelIdx);

    return this->rawData + channelOffset + (row * this->numOfCols) + col;
}

template <typename T>
inline T* Matrix<T>::operator()(const int& row, const int& col, const int& channelIdx, const int &sampleIdx) {
//    if (col >= this->numOfCols) {
//        throw std::invalid_argument("Matrix(row,col,channelIdx, sampleIdx) out of range: col >= this->numOfCols.");
//    } else if (row >= this->numOfRows) {
//        throw std::invalid_argument("Matrix(row,col,channelIdx, sampleIdx) out of range: row >= this->numOfRows.");
//    } else if (channelIdx >= this->depth) {
//        throw std::invalid_argument("Matrix(row,col,channelIdx, sampleIdx) out of range: channelIdx >= this->depth.");
//    } else if (sampleIdx >= this->batchSize) {
//      throw std::invalid_argument("Matrix(row,col,channelIdx, sampleIdx) out of range: sampleIdx >= this->batchSize.");
//    }

    auto offsetBySample = (this->numOfRows * this->numOfCols * this->depth * sampleIdx);
    auto channelOffset = (this->numOfRows * this->numOfCols * channelIdx);

    return this->rawData + offsetBySample + channelOffset + (row * this->numOfCols) + col;
}

template <typename T>
Matrix<T>* Matrix<T>::operator*(Matrix& other) {
    if (this->getNumOfCols() != other.getNumOfRows()) {
        throw std::invalid_argument("Matrices with incompatible dimension passed for multiplication.");
    }

    if (this->getDepth() > 1 || other.getDepth() > 1) {
      throw std::invalid_argument("Matrices with for multiplication greater depth than 1.");
    }

//    if (this->getBatchSize() > 1 || other.getBatchSize() > 1) {
//      throw std::invalid_argument("Matrices with for multiplication greater batch size than 1.");
//    }


  auto thisNumOfRows = this->getNumOfRows();
    auto thisNumOfCols = this->getNumOfCols();
    auto otherNumOfCols = other.getNumOfCols();

    auto resultMatrix = new Matrix(thisNumOfRows, otherNumOfCols, 1, other.getBatchSize());

    // A = (m,n)
    // B = (n,j)
    // C = A * B
    // C = (m,j)
    //
    //      A        *      B        =                                  C
    // |a11 ... a1n| * |b11 ... b1j| = | a11 * b11 + ... + a1n * bn1 | ... | a11 * b1j + ... + a1n * bnj |
    // [... ... ...|   [... ... ...|   |             ...             | ... |             ...             |
    // |am1 ... amn|   |bn1 ... bnj|   | am1 * b11 + ... + amn * bn1 | ... | am1 * b1j + ... + amn * bnj |

    // CACHE FRIENDLY IMPLEMENTATION OF MATRIX MULTIPLICATION
    // WE EXPERIMENTED WITH DIFFERENT TILE SIZES AND 50 SEEMS TO WORK GOOD
    const int TILE_SIZE = 50;
//    resultMatrix->setAllElementsZero();

//  for (auto batchIdx = 0; batchIdx < other.batchSize; batchIdx++) {
//    for (auto i = 0; i < thisNumOfRows; i += TILE_SIZE) {
//        const int maxI = std::min(i + TILE_SIZE, thisNumOfRows);
//
//        for (auto j = 0; j < otherNumOfCols; j+= TILE_SIZE) {
//            const int maxJ = std::min(j + TILE_SIZE, otherNumOfCols);
//
//            for (auto k = 0; k < thisNumOfCols; k+= TILE_SIZE) {
//                const int maxK = std::min(k + TILE_SIZE, thisNumOfCols);
//                for (auto i2 = i; i2 < maxI; i2++) {
//                    for (auto j2 = j; j2 < maxJ; j2++) {
//                        for (auto k2 = k; k2 < maxK; k2++) {
//                            *(*resultMatrix)(i2, j2, 0, batchSize) += *(*this)(i2, k2, 0, 0) * *other(k2, j2, 0, batchSize);
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//  }


//    resultMatrix->setAllElementsZero();

  for (auto batchIdx = 0; batchIdx < other.batchSize; batchIdx++) {
    for (auto resultRow = 0; resultRow < thisNumOfRows; resultRow++) {
      // Iterate over every column from Matrix B
      for (auto resultCol = 0; resultCol < otherNumOfCols; resultCol++) {
        T result = 0;
        // Perform sum of the dot product between two matching vectors from A and B
        for (auto sumIndex = 0; sumIndex < thisNumOfCols; sumIndex++) {
          // From A we take a row vector and from B we take a column vector
          result += *(*this)(resultRow, sumIndex, 0, 0) * *other(sumIndex, resultCol, 0, batchIdx);
        }

        *(*resultMatrix)(resultRow,resultCol, 0,batchIdx) = result;
      }
    }
  }


  return resultMatrix;
}
//
//template <typename T>
//Matrix<T>* Matrix<T>::matmul(Matrix& other) {
//    if (this->getNumOfCols() != other.getNumOfRows()) {
//        throw std::invalid_argument("Matrices with incompatible dimension passed for multiplication.");
//    }
//
//    if (this->getNumOfCols() != 1) {
//      throw std::invalid_argument("Matrices with dimension larger than 1 in cols passed to matmul!.");
//    }
//
//    if (this->getDepth() > 1 || other.getDepth() > 1) {
//      throw std::invalid_argument("Matrices with for multiplication greater depth than 1.");
//    }
//
////    if (this->getBatchSize() > 1 || other.getBatchSize() > 1) {
////      throw std::invalid_argument("Matrices with for multiplication greater batch size than 1.");
////    }
//
//
//    auto thisNumOfRows = this->getNumOfRows();
//    auto thisNumOfCols = this->getNumOfCols();
//    auto otherNumOfCols = other.getNumOfCols();
//
//    auto resultMatrix = new Matrix(thisNumOfRows, otherNumOfCols, 1, other.getBatchSize());
//
//    // A = (m,n)
//    // B = (n,j)
//    // C = A * B
//    // C = (m,j)
//    //
//    //      A        *      B        =                                  C
//    // |a11 ... a1n| * |b11 ... b1j| = | a11 * b11 + ... + a1n * bn1 | ... | a11 * b1j + ... + a1n * bnj |
//    // [... ... ...|   [... ... ...|   |             ...             | ... |             ...             |
//    // |am1 ... amn|   |bn1 ... bnj|   | am1 * b11 + ... + amn * bn1 | ... | am1 * b1j + ... + amn * bnj |
//
//    // CACHE FRIENDLY IMPLEMENTATION OF MATRIX MULTIPLICATION
//    // WE EXPERIMENTED WITH DIFFERENT TILE SIZES AND 50 SEEMS TO WORK GOOD
//    const int TILE_SIZE = 50;
////    resultMatrix->setAllElementsZero();
//
////  for (auto batchIdx = 0; batchIdx < other.batchSize; batchIdx++) {
////    for (auto i = 0; i < thisNumOfRows; i += TILE_SIZE) {
////        const int maxI = std::min(i + TILE_SIZE, thisNumOfRows);
////
////        for (auto j = 0; j < otherNumOfCols; j+= TILE_SIZE) {
////            const int maxJ = std::min(j + TILE_SIZE, otherNumOfCols);
////
////            for (auto k = 0; k < thisNumOfCols; k+= TILE_SIZE) {
////                const int maxK = std::min(k + TILE_SIZE, thisNumOfCols);
////                for (auto i2 = i; i2 < maxI; i2++) {
////                    for (auto j2 = j; j2 < maxJ; j2++) {
////                        for (auto k2 = k; k2 < maxK; k2++) {
////                            *(*resultMatrix)(i2, j2, 0, batchSize) += *(*this)(i2, k2, 0, 0) * *other(k2, j2, 0, batchSize);
////                        }
////                    }
////                }
////            }
////        }
////    }
////
////  }
//
//
////    resultMatrix->setAllElementsZero();
//
//  for (auto batchIdx = 0; batchIdx < other.batchSize; batchIdx++) {
//    for (auto resultRow = 0; resultRow < thisNumOfRows; resultRow++) {
//      // Iterate over every column from Matrix B
//      for (auto resultCol = 0; resultCol < otherNumOfCols; resultCol++) {
//        // From A we take a row vector and from B we take a column vector
//        *(*resultMatrix)(resultRow, resultCol, 0,batchIdx) = *(*this)(resultRow, 0, 0, 0) * *other(0, resultCol, 0, batchIdx);
//      }
//    }
//  }
//
//
//  return resultMatrix;
//}

template <typename T>
void Matrix<T>::operator-=(Matrix &other) {
  if (
    this->numOfRows != other.numOfRows
    || this->numOfCols != other.numOfCols
    || this->batchSize != other.batchSize
    || this->depth != other.depth) {
      throw std::invalid_argument("Invalid dimensions for matrix subtraction!");
  }

  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*this)(i, j, d, b) -= *(other(i, j, d, b));
        }
      }
    }
  }


}

template <typename T>
void Matrix<T>::operator+=(Matrix &other) {
  if (this->numOfRows != other.numOfRows
    || this->numOfCols != other.numOfCols
    || this->batchSize != other.batchSize
    || this->depth != other.depth) {
    throw std::invalid_argument("Invalid dimensions for matrix addition!");
  }

  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*this)(i, j, d, b) += *(other(i, j, d, b));
        }
      }
    }
  }
}

template <typename T>
void Matrix<T>::operator+=(T &scalar) {

  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*this)(i, j, d, b) += scalar;
        }
      }
    }
  }

}

template <typename T>
void Matrix<T>::operator/=(const T &scalar) {

  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*this)(i, j, d, b) /= scalar;
        }
      }
    }
  }

}

template <typename T>
Matrix<T> *Matrix<T>::operator-(Matrix &other) {
  if (this->numOfRows != other.numOfRows
      || this->numOfCols != other.numOfCols
      || this->batchSize != other.batchSize
      || this->depth != other.depth) {
    throw std::invalid_argument("Invalid dimensions for matrix subtraction!");
  }

  Matrix<double> *output = new Matrix<double>(numOfRows, numOfCols, depth, batchSize);

  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*output)(i, j, d, b) = *(*this)(i,j,d,b) - *(other)(i,j,d,b);
        }
      }
    }
  }


  return output;
}

template <typename T>
Matrix<T> *Matrix<T>::operator/(Matrix &other) {
  if (this->numOfRows != other.numOfRows
      || this->numOfCols != other.numOfCols
      || this->batchSize != other.batchSize
      || this->depth != other.depth) {
        throw std::invalid_argument("Invalid dimensions for matrix element wise division!");
    }

  Matrix<double> *output = new Matrix<double>(numOfRows, numOfCols, depth, batchSize);


  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*output)(i, j, d, b) = *(*this)(i,j,d,b) / *(other)(i,j,d,b);
        }
      }
    }
  }


  return output;
}


template <typename T>
Matrix<T> *Matrix<T>::operator*(const T &scalar) {
  Matrix<double> *output = new Matrix<double>(numOfRows, numOfCols, depth, batchSize);

//    for (auto i = 0; i < numOfRows; i++) {
//        for (auto j = 0; j < numOfCols; j++) {
//            (*output)[i][j] = (*this)[i][j] * scalar;
//        }
//    }


  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*output)(i, j, d, b) = *(*this)(i, j, d, b) * scalar;
        }
      }
    }
  }


  return output;
}

template <typename T>
void Matrix<T>::operator*=(const T &scalar) {
  Matrix<double> *output = this;

  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*output)(i, j, d, b) *= scalar;
        }
      }
    }
  }

}

/*********** METHODS ********************************************************/

template <typename T>
void Matrix<T>::copyElementsFrom(Matrix<T> &other, bool copyMemory) {
    if (this->numOfRows != other.numOfRows
      || this->numOfCols != other.numOfCols
      || this->depth != other.depth
      || this->batchSize != other.batchSize) {
        throw std::invalid_argument("Invalid dimensions for matrix copy!");
    }

    if (copyMemory) {
      memcpy(this->rawData, other.rawData, sizeof(T) * numOfRows * numOfCols * depth * batchSize);
    } else {
      this->rawData = other.rawData;
      this->notOwningData = true;
    }
}

template <typename T>
Matrix<T>* Matrix<T>::copy() {
  auto newMatrix = new Matrix(rawData, numOfRows, numOfCols, true, depth, batchSize);

  return newMatrix;
}

template <typename T>
Matrix<T> *Matrix<T>::reshape(int newNumOfRows, int newNumOfCols, int newDepth, int newBatchSize){
  auto currentSize = this->numOfRows * this->numOfCols * this->depth * this->batchSize;
  auto newSize = newNumOfRows * newNumOfCols * newDepth * newBatchSize;

  if (newSize > currentSize) {
    std::stringstream ss;

    ss << "Invalid matrix.reshape()! "
       << "Cannot reshape " << "["
       << this->numOfRows << "],["
       << this->numOfCols << "],["
       << this->depth     << "],["
       << this->batchSize << "]"
       << " into" << "["
       << newNumOfRows << "],["
       << newNumOfCols << "],["
       << newDepth     << "],["
       << newBatchSize << "]: "
       << currentSize << " != "
       << newSize << std::endl;

    throw std::invalid_argument(ss.str());
  }

  this->numOfRows = newNumOfRows;
  this->numOfCols = newNumOfCols;
  this->depth = newDepth;
  this->batchSize = newBatchSize;

  return this;
}

template <typename T>
Matrix<T> *Matrix<T>::sqrt() {

  Matrix<double> *output = new Matrix<double>(numOfRows, numOfCols, depth, batchSize);


  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*output)(i, j, d, b) = std::sqrt(*(*this)(i,j,d,b));
        }
      }
    }
  }


  return output;
}

template <typename T>
Matrix<T> *Matrix<T>::pow(double exponent) {
  Matrix<double> *output = new Matrix<double>(numOfRows, numOfCols, depth, batchSize);

  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*output)(i, j, d, b) = std::pow(*(*this)(i,j,d,b), exponent);
        }
      }
    }
  }


  return output;
}

template <typename T>
T Matrix<T>::totalAbsDifference(Matrix<T> *other) {
  T diff = 0;
  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          diff += std::fabs(*(*this)(i,j,d,b) - *(*other)(i,j,d,b));
        }
      }
    }
  }

  return diff;
}

template <typename T>
T Matrix<T>::absoluteSum() {
  T sum = 0;
  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          sum += std::fabs(*(*this)(i,j,d,b));
        }
      }
    }
  }

  return sum;
}
template <typename T>
Matrix<T> *Matrix<T>::componentWiseReluDerivMult(Matrix *reluOutputs, bool inPlace) {
  if (this->numOfRows != reluOutputs->numOfRows
      || this->numOfCols != reluOutputs->numOfCols
      || this->batchSize != reluOutputs->batchSize
      || this->depth != reluOutputs->depth) {
    throw std::invalid_argument("Invalid dimensions for matrix component wise multiplication!");
  }

  Matrix<T> *output;
  if (inPlace) {
    output = this;
  } else {
    output = new Matrix<double>(numOfRows, numOfCols, depth, batchSize);
  }

  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {

          auto value = 0.0;
          if (*(*reluOutputs)(i,j,d,b) > 0 ){
            value = 1.0;
          }

          *(*output)(i, j, d, b) = *(*this)(i,j,d,b) * value;
        }
      }
    }
  }


  return output;
}

template <typename T>
Matrix<T> *Matrix<T>::componentWiseMult(Matrix *other, bool inPlace) {
  if (this->numOfRows != other->numOfRows
      || this->numOfCols != other->numOfCols
      || this->batchSize != other->batchSize
      || this->depth != other->depth) {
    throw std::invalid_argument("Invalid dimensions for matrix component wise multiplication!!!");
  }

  Matrix<T> *output;
  if (inPlace) {
    output = this;
  } else {
    output = new Matrix<double>(numOfRows, numOfCols, depth, batchSize);
  }

  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*output)(i, j, d, b) = *(*this)(i,j,d,b) * *(*other)(i,j,d,b);
        }
      }
    }
  }


  return output;
}



template <typename T>
void Matrix<T>::printValues() {
    std::cout << std::endl;

    for (auto b = 0; b < batchSize; b++) {
      for (auto z = 0; z < depth; z++) {
        for (auto i = 0; i < numOfRows; i++) {
          for (auto j = 0; j < numOfCols; j++) {
            std::cout << *(*this)(i, j, z, b) << " ";
          }
          std::cout << std::endl;
        }

        if (z != depth - 1) {
          std::cout << std::endl;
          std::cout << std::endl;
        }
      }

      if (b != batchSize -1) {
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
      }
    }

}


template <typename T>
inline int Matrix<T>::getNumOfRows() const {
    return this->numOfRows;
}

template <typename T>
inline int Matrix<T>::getNumOfCols() const {
    return this->numOfCols;
}

template <typename T>
std::string Matrix<T>::toString(){
    std::stringstream ss;

    for (auto i = 0; i < numOfRows; i++) {
        for(auto j = 0; j < numOfCols; j++) {
            ss << (*this)[i][j] << " ";
        }
        ss << std::endl;
    }
    return ss.str();
}

template <typename T>
void Matrix<T>::updateDebug(){
    if (this->values.size() == 0) {
        this->values.clear();
        this->values.reserve(numOfRows * numOfCols);
        for (auto i = 0; i < numOfRows; i++) {
            for(auto j = 0; j < numOfCols; j++) {
                this->values.push_back((*this)[i][j]);
            }
        }
    } else {
        for (auto i = 0; i < numOfRows; i++) {
            for(auto j = 0; j < numOfCols; j++) {
                this->values[(i*numOfCols) + j] = (*this)[i][j];
            }
        }
    }
}

template <typename T>
Matrix<T>* Matrix<T>::transposeToNew(){
    T *newArrayData = (T *) malloc (sizeof(T) * this->getNumOfRows() * this->getNumOfCols());

    auto res = new Matrix<T>(this->getNumOfCols(), this->getNumOfRows(), this->getDepth(), this->getBatchSize());


    for (auto i = 0; i < this->getNumOfRows(); i++) {
        for (auto j = 0; j < this->getNumOfCols(); j++) {
            auto value = (*this)[i][j];

            newArrayData[(j * this->getNumOfRows()) + i] = value;
        }
    }

    return new Matrix<T>(newArrayData, this->getNumOfCols(), this->getNumOfRows(), false, this->getDepth(), this->getBatchSize());
}

template <typename T>
void Matrix<T>::setAllElementsZero() {
  auto numOfBytes = sizeof(T) * this->getNumOfCols() * this->getNumOfRows() * this->getDepth() * this->getBatchSize();
  if (numOfBytes <= 0) {
    throw std::invalid_argument("Setting 0 bytes to zero!!!");
  }
  memset(this->rawData, 0, numOfBytes);
//
//  for (auto b = 0; b < batchSize; b++) {
//    for (auto d = 0; d < depth; d++) {
//      for (auto i = 0; i < numOfRows; i++) {
//        for (auto j = 0; j < numOfCols; j++) {
//          *(*this)(i,j,d,b) = 0;
//        }
//      }
//    }
//  }
}

template <typename T>
void Matrix<T>::setAllElementsTo(const T &value) {
  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*this)(i,j,d,b) = value;
        }
      }
    }
  }
}


template <typename T>
Matrix<T>* Matrix<T>::softmax() {
    auto &matrix = *this;

//    T *data = (T *) malloc(matrix.getNumOfRows() * matrix.getBatchSize() * sizeof(T));

    auto result = new Matrix(matrix.getNumOfRows(), 1, 1, matrix.getBatchSize());
    result->setAllElementsZero();

    // TODO: probably transpose matrix (so we don't jump across rows when performing softmax for one column...)
    for (auto j = 0; j < matrix.getBatchSize(); j++) {
        auto max = *matrix(0, 0, 0, j);

        for (auto i = 1; i < matrix.getNumOfRows(); i++) {
            if (*matrix(i,0,0,j) > max) {
                max = *matrix(i,0,0,j);
            }
        }

        T rowSum = 0.0;
        for (auto i = 0; i < matrix.getNumOfRows(); i++) {
//            data[(i * matrix.getNumOfCols()) + j] = std::exp(matrix[i][j] - max);
            *(*result)(i, 0, 0, j) = std::exp(*matrix(i,0,0,j) - max);
            rowSum += *(*result)(i, 0, 0, j);
        }

        T rowSum2 = 0.0;
        for (auto i = 0; i < matrix.getNumOfRows(); i++) {
            rowSum2 += *(*result)(i, 0, 0, j);
        }

        for (auto i = 0; i < matrix.getNumOfRows(); i++) {
            *(*result)(i, 0, 0, j) = *(*result)(i, 0, 0, j) / rowSum2;
        }

        T normalizedRowSum = 0.0;
        for (auto i = 0; i < matrix.getNumOfRows(); i++) {
            normalizedRowSum += *(*result)(i, 0, 0, j);
        }

        if (std::fabs(normalizedRowSum - 1.0) > 0.00001) {
          throw std::invalid_argument("Normalized row sum != 1.0 in softmax!!!");
        }

    }

    return result;
}

template <typename T>
Matrix<T>* Matrix<T>::softmaxInPlace() {
    auto &matrix = *this;

    // TODO: probably transpose matrix (so we don't jump across rows when performing softmax for one column...)
    for (auto j = 0; j < matrix.getNumOfCols(); j++) {
        auto max = matrix[0][j];

        for (auto i = 1; i < matrix.getNumOfRows(); i++) {
            if (matrix[i][j] > max) {
                max = matrix[i][j];
            }
        }

        T rowSum = 0.0;
        for (auto i = 0; i < matrix.getNumOfRows(); i++) {
            auto shiftedValue = (long double) matrix[i][j] - max;
            rowSum += std::exp(shiftedValue);
            matrix[i][j] = std::exp(shiftedValue);
        }

        for (auto i = 0; i < matrix.getNumOfRows(); i++) {
            matrix[i][j] /= rowSum;
        }
    }

//    matrix.updateDebug();

    return this;
}


template <typename T>
Matrix<T>* Matrix<T>::relu() {
  auto output = new Matrix(this->getNumOfRows(), this->getNumOfCols(), this->getDepth(), this->getBatchSize());


  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*output)(i, j, d, b) = std::fmax(*(*this)(i,j,d,b), 0);
        }
      }
    }
  }

  return output;
}

template <typename T>
Matrix<T>* Matrix<T>::reluInPlace() {
  auto output = this;

  for (auto b = 0; b < batchSize; b++) {
    for (auto d = 0; d < depth; d++) {
      for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
          *(*output)(i, j, d, b) = std::fmax(*(*this)(i, j, d, b), 0);
        }
      }
    }
  }

  return this;
}

template <typename T>
Matrix<T>* Matrix<T>::argMaxByRow() {
    auto &matrix = *this;

    auto newMatrix = new Matrix(this->getBatchSize(), 1);


    for (auto j = 0; j < matrix.getBatchSize(); j++) {
        auto rowMax = *(matrix(0, 0, 0, j));
        auto rowMaxArg = 0;

        for (auto i = 1; i < matrix.getNumOfRows(); i++) {
            if (*matrix(i, 0, 0, j) > rowMax) {
                rowMax = *matrix(i, 0, 0, j);
                rowMaxArg = i;
            }
        }

        (*newMatrix)[j][0] = rowMaxArg;
    }

    return newMatrix;
}


// TODO: This is somewhat numerically unstable
template <typename T>
T Matrix<T>::crossEntropyLoss(Matrix<T> *true_values) {
    auto &matrix = *this;

    T loss = 0;
    for (auto j = 0; j < matrix.getNumOfCols(); j++) {
        for (auto i = 0; i < matrix.getNumOfRows(); i++) {
            if ((*true_values)[i][j] != 0) {
                if (loss == 0) {
                    loss = matrix[i][j];
                } else if(matrix[i][j] != 0) {
                    loss *= matrix[i][j];
                } else if(matrix[i][j] == 0) {
                    loss *= 1e-32;
                }
            }
        }
    }

    loss = -1 * log(std::max(loss, 1e-32));

    return loss;
}

template class Matrix<double>;
