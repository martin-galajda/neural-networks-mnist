// Created by Martin Galajda on 20/10/2018.
//

#include <iostream>
#include <cmath>

#ifndef PV021_MATRIX_IMPL_MatrixV6_H
#define PV021_MATRIX_IMPL_MatrixV6_H

template <typename T>
class RowVector {
public:
    RowVector(T *rowData);
    ~RowVector();

    T& operator[](const int& index);

private:
    T *rowData;
};


template <typename T>
class MatrixV6 {
public:

    MatrixV6(int rows, int cols);
    MatrixV6(T* array, int rows, int cols);
    MatrixV6(T* array, int rows, int cols, bool copyData);
    ~MatrixV6();

    RowVector<T>& operator[](const int& index);
    MatrixV6* operator*(MatrixV6&);

    MatrixV6* transposeToNew();
    MatrixV6* softmax();

    int getNumOfRows();
    int getNumOfCols();
    void printValues();
    std::string toString();


protected:
    RowVector<T>** rows;
    T *rawData;

    int numOfRows;
    int numOfCols;
};



/*********** CONSTRUCTORS ********************************************************/

template <typename T>
MatrixV6<T>::MatrixV6(int rows, int cols) {
    this->numOfRows = rows;
    this->numOfCols = cols;
    this->rows = (RowVector<T>**) malloc(sizeof(RowVector<T>*) * rows);

    this->rawData = (T*) malloc(sizeof(T) * rows * cols);
    for (auto i = 0; i < rows; i++) {
        RowVector<T> *row = new RowVector<T>(rawData + (cols * i));
        this->rows[i] = row;
    }
}

template <typename T>
MatrixV6<T>::MatrixV6(T *data, int rows, int cols): MatrixV6(rows, cols) {
    this->numOfRows = rows;
    this->numOfCols = cols;

    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            T arrayValue = *(data + (i * numOfCols) + j);
            RowVector<T>& row = *(*(this->rows + i));

            T& value = row[j];
            value = arrayValue;
        }
    }
}

template <typename T>
MatrixV6<T>::MatrixV6(T *data, int rows, int cols, bool copyData) {
    this->numOfRows = rows;
    this->numOfCols = cols;

    this->rows = (RowVector<T>**) malloc(sizeof(RowVector<T>*) * rows);

    for (auto i = 0; i < numOfRows; i++) {
        *(this->rows + i) = new RowVector(data + (i * numOfCols));
    }
}

template <typename T>
RowVector<T>::RowVector(T *rowData) {
    this->rowData = rowData;
}

template <typename T>
RowVector<T>::~RowVector() {}

/*********** DESTRUCTORS ********************************************************/

template <typename T>
MatrixV6<T>::~MatrixV6() {
    for (auto i = 0; i < numOfRows; i++) {
        RowVector<T>* vectorPtr = *(this->rows + i);

        free(vectorPtr);
    }
    free(rows);
    free(rawData);
}

/*********** OPERATORS ********************************************************/

template <typename T>
T& RowVector<T>::operator[](const int& index) {
    T& elementRef = *(this->rowData + index);
    return elementRef;
}


template <typename T>
RowVector<T> &MatrixV6<T>::operator[](const int& index) {
    return *(*(this->rows + index));
}

template <typename T>
MatrixV6<T>* MatrixV6<T>::operator*(MatrixV6& other) {
    if (this->getNumOfCols() != other.getNumOfRows()) {
        throw std::invalid_argument("Matrices with incompatible dimension passed for multiplication.");
    }

    auto thisNumOfRows = this->getNumOfRows();
    auto thisNumOfCols = this->getNumOfCols();
    auto otherNumOfCols = other.getNumOfCols();

    auto resultMatrixV6 = new MatrixV6(thisNumOfRows, otherNumOfCols);

    // A = (m,n)
    // B = (n,j)
    // C = A * B
    // C = (m,j)
    //
    //      A        *      B        =                                  C
    // |a11 ... a1n| * |b11 ... b1j| = | a11 * b11 + ... + a1n * bn1 | ... | a11 * b1j + ... + a1n * bnj |
    // [... ... ...|   [... ... ...|   |             ...             | ... |             ...             |
    // |am1 ... amn|   |bn1 ... bnj|   | am1 * b11 + ... + amn * bn1 | ... | am1 * b1j + ... + amn * bnj |

    auto *otherTransposed = other.transposeToNew();

    // Iterate over every row from MatrixV6 A
    for (auto resultRow = 0; resultRow < thisNumOfRows; resultRow++) {
        // Iterate over every column from MatrixV6 B
        for (auto resultCol = 0; resultCol < otherNumOfCols; resultCol++) {
            T result = 0;
            // Perform sum of the dot product between two matching vectors from A and B
            for (auto sumIndex = 0; sumIndex < thisNumOfCols; sumIndex++) {
                // From A we take a row vector and from B we take a column vector
                result += (*this)[resultRow][sumIndex] * (*otherTransposed)[resultCol][sumIndex];
            }

            (*resultMatrixV6)[resultRow][resultCol] = result;
        }
    }

    free(otherTransposed);

    return resultMatrixV6;
}

/*********** METHODS ********************************************************/

template <typename T>
void MatrixV6<T>::printValues() {
    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            std::cout << this->operator[](i)[j] << " ";
        }
        std::cout << std::endl;
    }
}


template <typename T>
int MatrixV6<T>::getNumOfRows() {
    return this->numOfRows;
}

template <typename T>
int MatrixV6<T>::getNumOfCols() {
    return this->numOfCols;
}

template <typename T>
std::string MatrixV6<T>::toString(){
    std::string result = "";

    for (auto i = 0; i < numOfRows(); i++) {
        for(auto j = 0; j < numOfCols(); j++) {
            result += (*this)[i][j] + " ";
        }
        result += '\n';
    }
    return result;
}

template <typename T>
MatrixV6<T>* MatrixV6<T>::transposeToNew(){
    T *newArrayData = (T *) malloc (sizeof(T) * this->getNumOfRows() * this->getNumOfCols());

    for (auto i = 0; i < this->getNumOfRows(); i++) {
        for (auto j = 0; j < this->getNumOfCols(); j++) {
            auto value = (*this)[i][j];
            newArrayData[(j * this->getNumOfRows()) + i] = value;
        }
    }

    return new MatrixV6<T>(newArrayData, this->getNumOfCols(), this->getNumOfRows(), false);
}

template <typename T>
MatrixV6<T>* MatrixV6<T>::softmax() {
    auto &matrix = *this;

    T *data = (T *) malloc(matrix.getNumOfRows() * matrix.getNumOfCols() * sizeof(T));

    // TODO: probably transpose matrix (so we don't jump across rows when performing softmax for one column...)
    for (auto j = 0; j < matrix.getNumOfCols(); j++) {
        auto col = matrix[j];
        auto max = col[0];

        for (auto i = 1; i < matrix.getNumOfRows(); i++) {
            if (matrix[i][j] > max) {
                max = matrix[i][j];
            }
        }

        T rowSum = 0;
        for (auto i = 0; i < matrix.getNumOfRows(); i++) {
            data[(i * matrix.getNumOfCols()) + j] = std::exp(matrix[i][j] - max);
            rowSum += std::exp(matrix[i][j] - max);
        }

        for (auto i = 0; i < matrix.getNumOfRows(); i++) {
            data[(i * matrix.getNumOfCols()) + j] /= rowSum;
        }
    }

    return new MatrixV6(data, matrix.getNumOfRows(), matrix.getNumOfCols());
}


#endif //PV021_MATRIX_IMPL_MatrixV6_H
