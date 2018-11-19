//
// Created by Martin Galajda on 25/10/2018.
//
#define _GLIBCXX_USE_CXX11_ABI 0

#include <cmath>
#include <cstring>
#include "./Matrix.hpp"
#include <sstream>


/*********** CONSTRUCTORS ********************************************************/

template <typename T>
Matrix<T>::Matrix(int rows, int cols) {
    this->numOfRows = rows;
    this->numOfCols = cols;
//    this->rows = (RowVector<T>**) malloc(sizeof(RowVector<T>*) * rows);
//
    this->rawData = (T*) malloc(sizeof(T) * rows * cols);
//    for (auto i = 0; i < numOfRows; i++) {
//        RowVector<T> *row = new RowVector<T>(rawData + (numOfCols * i));
//        this->rows[i] = row;
//    }
}

template <typename T>
Matrix<T>::Matrix(int rows, int cols, BaseInitializer *initializer): Matrix(rows, cols) {

    for (auto row = 0; row < numOfRows; row++) {
        for (auto col = 0; col < numOfCols; col++) {
            (*this)[row][col] = initializer->getValue();
        }
    }

}

template <typename T>
Matrix<T>::Matrix(T *data, int rows, int cols): Matrix(rows, cols) {
    this->numOfRows = rows;
    this->numOfCols = cols;

    memcpy(this->rawData, data, sizeof(T) * rows * cols);
}

template <typename T>
Matrix<T>::Matrix(T *data, int rows, int cols, bool copyData): Matrix(rows, cols) {
    this->numOfRows = rows;
    this->numOfCols = cols;


//    this->rows = (RowVector<T>**) malloc(sizeof(RowVector<T>*) * rows);

//    memcpy(this->rawData, data, sizeof(T) * rows * cols);
    this->rawData = data;

//    for (auto i = 0; i < numOfRows; i++) {
////        *(this->rows + i) = new RowVector(data + (i * numOfCols));
//    }
}

template <typename T>
RowVector<T>::RowVector(T *rowData) {
    this->rowData = rowData;
}

template <typename T>
RowVector<T>::~RowVector() {

}

/*********** DESTRUCTORS ********************************************************/

template <typename T>
Matrix<T>::~Matrix() {
    free(rawData);
}

/*********** OPERATORS ********************************************************/

template <typename T>
inline T* Matrix<T>::operator[](const int& index) {
    return this->rawData + (index * numOfCols);
}




template <typename T>
Matrix<T>* Matrix<T>::operator*(Matrix& other) {
    if (this->getNumOfCols() != other.getNumOfRows()) {
        throw std::invalid_argument("Matrices with incompatible dimension passed for multiplication.");
    }

    auto thisNumOfRows = this->getNumOfRows();
    auto thisNumOfCols = this->getNumOfCols();
    auto otherNumOfCols = other.getNumOfCols();

    auto resultMatrix = new Matrix(thisNumOfRows, otherNumOfCols);

    // A = (m,n)
    // B = (n,j)
    // C = A * B
    // C = (m,j)
    //
    //      A        *      B        =                                  C
    // |a11 ... a1n| * |b11 ... b1j| = | a11 * b11 + ... + a1n * bn1 | ... | a11 * b1j + ... + a1n * bnj |
    // [... ... ...|   [... ... ...|   |             ...             | ... |             ...             |
    // |am1 ... amn|   |bn1 ... bnj|   | am1 * b11 + ... + amn * bn1 | ... | am1 * b1j + ... + amn * bnj |

//    auto *otherTransposed = other.transposeToNew();
//
//    // Iterate over every row from Matrix A
//    for (auto resultRow = 0; resultRow < thisNumOfRows; resultRow++) {
//        // Iterate over every column from Matrix B
//        for (auto resultCol = 0; resultCol < otherNumOfCols; resultCol++) {
//            T result = 0;
//            // Perform sum of the dot product between two matching vectors from A and B
//            for (auto sumIndex = 0; sumIndex < thisNumOfCols; sumIndex++) {
//                // From A we take a row vector and from B we take a column vector
//                result += (*this)[resultRow][sumIndex] * (*otherTransposed)[resultCol][sumIndex];
//            }
//
//            (*resultMatrix)[resultRow][resultCol] = result;
//        }
//    }
//
//    free(otherTransposed);


    // CACHE FRIENDLY IMPLEMENTATION OF MATRIX MULTIPLICATION
    // WE EXPERIMENTED WITH DIFFERENT TILE SIZES AND 50 SEEMS TO WORK GOOD
    const int TILE_SIZE = 50;
    resultMatrix->setAllElementsZero(0.0);

    for (auto i = 0; i < thisNumOfRows; i += TILE_SIZE) {
        const int maxI = std::min(i + TILE_SIZE, thisNumOfRows);

        for (auto j = 0; j < otherNumOfCols; j+= TILE_SIZE) {
            const int maxJ = std::min(j + TILE_SIZE, otherNumOfCols);

            for (auto k = 0; k < thisNumOfCols; k+= TILE_SIZE) {
                const int maxK = std::min(k + TILE_SIZE, thisNumOfCols);
                for (auto i2 = i; i2 < maxI; i2++) {
                    for (auto j2 = j; j2 < maxJ; j2++) {
                        for (auto k2 = k; k2 < maxK; k2++) {
                            (*resultMatrix)[i2][j2] += (*this)[i2][k2] * other[k2][j2];
                        }
                    }
                }
            }
        }
    }


    return resultMatrix;
}

template <typename T>
void Matrix<T>::operator-=(Matrix &other) {
    if (this->numOfRows != other.numOfRows || this->numOfCols != other.numOfCols) {
        throw std::invalid_argument("Invalid dimensions for matrix subtraction!");
    }

    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            (*this)[i][j] -= other[i][j];
        }
    }
}

template <typename T>
void Matrix<T>::operator+=(Matrix &other) {
    if (this->numOfRows != other.numOfRows || this->numOfCols != other.numOfCols) {
        throw std::invalid_argument("Invalid dimensions for matrix addition!");
    }

    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            (*this)[i][j] += other[i][j];
        }
    }
}

template <typename T>
void Matrix<T>::operator+=(T &scalar) {
    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            (*this)[i][j] += scalar;
        }
    }
}

template <typename T>
void Matrix<T>::operator/=(const T &scalar) {

    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            (*this)[i][j] /= scalar;
        }
    }
}

template <typename T>
Matrix<T> *Matrix<T>::operator-(Matrix &other) {
    if (this->numOfRows != other.numOfRows || this->numOfCols != other.numOfCols) {
        throw std::invalid_argument("Invalid dimensions for matrix subtraction!");
    }

    Matrix<double> *output = new Matrix<double>(numOfRows, numOfCols);

    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            (*output)[i][j] = (*this)[i][j] - other[i][j];
        }
    }

    return output;
}

template <typename T>
Matrix<T> *Matrix<T>::operator/(Matrix &other) {
    if (this->numOfRows != other.numOfRows || this->numOfCols != other.numOfCols) {
        throw std::invalid_argument("Invalid dimensions for matrix element wise division!");
    }

    Matrix<double> *output = new Matrix<double>(numOfRows, numOfCols);

    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            (*output)[i][j] = (*this)[i][j] / other[i][j];
        }
    }

    return output;
}


template <typename T>
Matrix<T> *Matrix<T>::operator*(const T &scalar) {
    Matrix<double> *output = new Matrix<double>(numOfRows, numOfCols);

    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            (*output)[i][j] = (*this)[i][j] * scalar;
        }
    }


    return output;
}

template <typename T>
void Matrix<T>::operator*=(const T &scalar) {
    Matrix<double> *output = this;

    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            (*output)[i][j] *= scalar;
        }
    }
}


/*********** METHODS ********************************************************/

template <typename T>
void Matrix<T>::copyElementsFrom(Matrix<T> &other) {
    if (this->numOfRows != other.numOfRows || this->numOfCols != other.numOfCols) {
        throw std::invalid_argument("Invalid dimensions for matrix copy!");
    }


    for (auto i = 0; i < this->getNumOfRows(); i++) {
        for (auto j = 0; j < this->getNumOfCols(); j++) {
            (*this)[i][j] = other[i][j];
        }
    }
}

template <typename T>
Matrix<T> *Matrix<T>::sqrt() {

    Matrix<double> *output = new Matrix<double>(numOfRows, numOfCols);

    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            (*output)[i][j] = std::sqrt((*this)[i][j]);
        }
    }

    return output;
}

template <typename T>
Matrix<T> *Matrix<T>::pow(double exponent) {

    Matrix<double> *output = new Matrix<double>(numOfRows, numOfCols);

    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            (*output)[i][j] = std::pow((*this)[i][j], exponent);
        }
    }

    return output;
}



template <typename T>
void Matrix<T>::printValues() {
    std::cout << std::endl;
    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            std::cout << (*this)[i][j] << " ";
        }
        std::cout << std::endl;
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
//    this->debugValues = this->toString();

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

    for (auto i = 0; i < this->getNumOfRows(); i++) {
        for (auto j = 0; j < this->getNumOfCols(); j++) {
            auto value = (*this)[i][j];
            newArrayData[(j * this->getNumOfRows()) + i] = value;
        }
    }

    return new Matrix<T>(newArrayData, this->getNumOfCols(), this->getNumOfRows(), false);
}

template <typename T>
void Matrix<T>::setAllElementsZero(const T &value) {
//    for (auto i = 0; i < this->getNumOfRows(); i++) {
//        for (auto j = 0; j < this->getNumOfCols(); j++) {
//            (*this)[i][j] = value;
//        }
//    }

    memset(this->rawData, 0, sizeof(T) * this->getNumOfCols() * this->getNumOfRows());
}


template <typename T>
Matrix<T>* Matrix<T>::softmax() {
    auto &matrix = *this;

    T *data = (T *) malloc(matrix.getNumOfRows() * matrix.getNumOfCols() * sizeof(T));

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
            data[(i * matrix.getNumOfCols()) + j] = std::exp(matrix[i][j] - max);
            rowSum += data[(i * matrix.getNumOfCols()) + j];
        }

        T rowSum2 = 0.0;
        for (auto i = 0; i < matrix.getNumOfRows(); i++) {
            rowSum2 += data[(i * matrix.getNumOfCols()) + j];
        }

        for (auto i = 0; i < matrix.getNumOfRows(); i++) {
            data[(i * matrix.getNumOfCols()) + j] = data[(i * matrix.getNumOfCols()) + j] / rowSum2;
        }

        T normalizedRowSum = 0.0;
        for (auto i = 0; i < matrix.getNumOfRows(); i++) {
            normalizedRowSum += data[(i * matrix.getNumOfCols()) + j];
        }

    }

    return new Matrix(data, matrix.getNumOfRows(), matrix.getNumOfCols(), true);
}

template <typename T>
Matrix<T>* Matrix<T>::softmaxInPlace() {
    auto &matrix = *this;

//    matrix.updateDebug();

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
    auto &matrix = *this;

    T *data = (T *) malloc(matrix.getNumOfRows() * matrix.getNumOfCols() * sizeof(T));

    for (auto j = 0; j < matrix.getNumOfCols(); j++) {
        for (auto i = 0; i < matrix.getNumOfRows(); i++) {
            if (matrix[i][j] < 0) {
                data[(i * matrix.getNumOfCols()) + j] = 0;
            } else {
                data[(i * matrix.getNumOfCols()) + j] = matrix[i][j];
            }
        }
    }

    return new Matrix(data, matrix.getNumOfRows(), matrix.getNumOfCols());
}

template <typename T>
Matrix<T>* Matrix<T>::reluInPlace() {
    auto &matrix = *this;

    for (auto i = 0; i < matrix.getNumOfRows(); i++) {
        for (auto j = 0; j < matrix.getNumOfCols(); j++) {
            if (matrix[i][j] < 0) {
                matrix[i][j] = 0;
            }
        }

    }

    return this;
}

template <typename T>
Matrix<T>* Matrix<T>::argMaxByRow() {
    auto &matrix = *this;

    auto newMatrix = new Matrix(this->getNumOfCols(), 1);


    for (auto j = 0; j < matrix.getNumOfCols(); j++) {
        auto rowMax = matrix[0][j];
        auto rowMaxArg = 0;

        for (auto i = 1; i < matrix.getNumOfRows(); i++) {
            if (matrix[i][j] > rowMax) {
                rowMax = matrix[i][j];
                rowMaxArg = i;
            }
        }

        (*newMatrix)[j][0] = rowMaxArg;
    }

    return newMatrix;
}

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
