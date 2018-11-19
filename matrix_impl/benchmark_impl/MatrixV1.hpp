//
// Created by Martin Galajda on 16/10/2018.
//

#ifndef PV021_MATRIX_IMPL_MatrixV1_H
#define PV021_MATRIX_IMPL_MatrixV1_H

#include <iostream>

template <typename T>
class MatrixV1 {
public:
    class RowVector {
    public:
        RowVector(T *rowData);
        ~RowVector();

        T& operator[](const int& index);

    private:
        T *rowData;
    };

    MatrixV1(int rows, int cols);
    MatrixV1(T* array, int rows, int cols);

    ~MatrixV1();

    MatrixV1::RowVector& operator[](const int& index);
    MatrixV1* operator*(MatrixV1&);

    int getNumOfRows();
    int getNumOfCols();
    void printValues();
protected:
    RowVector** rows;

    int numOfRows;
    int numOfCols;
};


/*********** CONSTRUCTORS ********************************************************/

template <typename T>
MatrixV1<T>::MatrixV1(int rows, int cols) {
    this->numOfRows = rows;
    this->numOfCols = cols;
    this->rows = (RowVector**) malloc(sizeof(RowVector*) * rows);
    for (auto i = 0; i < rows; i++) {
        RowVector *row = new MatrixV1::RowVector((T *) malloc(sizeof(T) * cols));
        this->rows[i] = row;
    }
}

template <typename T>
MatrixV1<T>::MatrixV1(T *data, int rows, int cols): MatrixV1(rows, cols) {
    this->numOfRows = rows;
    this->numOfCols = cols;

    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            T arrayValue = *(data + (i * numOfCols) + j);
            RowVector& row = *(*(this->rows + i));

            T& value = row[j];
            value = arrayValue;
        }
    }
}

template <typename T>
MatrixV1<T>::RowVector::RowVector(T *rowData) {
    this->rowData = rowData;
}

/*********** DESTRUCTORS ********************************************************/

template <typename T>
MatrixV1<T>::RowVector::~RowVector() {
    free(this->rowData);
}

template <typename T>
MatrixV1<T>::~MatrixV1() {
    for (auto i = 0; i < numOfRows; i++) {
        RowVector* vectorPtr = *(this->rows + i);

        free(vectorPtr);
    }
    free(rows);
}

/*********** OPERATORS ********************************************************/

template <typename T>
MatrixV1<T>* MatrixV1<T>::operator*(MatrixV1& other) {
    if (this->getNumOfCols() != other.getNumOfRows()) {
        throw std::invalid_argument("Matrices with incompatible dimension passed for multiplication.");
    }

    auto thisNumOfRows = this->getNumOfRows();
    auto thisNumOfCols = this->getNumOfCols();
    auto otherNumOfCols = other.getNumOfCols();

    auto resultMatrix = new MatrixV1(thisNumOfRows, otherNumOfCols);

    // A = (m,n)
    // B = (n,j)
    // C = A * B
    // C = (m,j)
    //
    //      A        *      B        =                                  C
    // |a11 ... a1n| * |b11 ... b1j| = | a11 * b11 + ... + a1n * bn1 | ... | a11 * b1j + ... + a1n * bnj |
    // [... ... ...|   [... ... ...|   |             ...             | ... |             ...             |
    // |am1 ... amn|   |bn1 ... bnj|   | am1 * b11 + ... + amn * bn1 | ... | am1 * b1j + ... + amn * bnj |

    // Iterate over every row from Matrix A
    for (auto resultRow = 0; resultRow < thisNumOfRows; resultRow++) {
        // Iterate over every column from Matrix B
        for (auto resultCol = 0; resultCol < otherNumOfCols; resultCol++) {
            T result = 0;
            // Perform sum of the dot product between two matching vectors from A and B
            for (auto sumIndex = 0; sumIndex < thisNumOfCols; sumIndex++) {
                // From A we take a row vector and from B we take a column vector
                result += (*this)[resultRow][sumIndex] * other[sumIndex][resultCol];
            }

            (*resultMatrix)[resultRow][resultCol] = result;
        }
    }

    return resultMatrix;
}


template <typename T>
T& MatrixV1<T>::RowVector::operator[](const int& index) {
    T& elementRef = *(this->rowData + index);
    return elementRef;
}

template <typename T>
typename MatrixV1<T>::RowVector& MatrixV1<T>::operator[](const int& index) {
    return *(*(this->rows + index));
}

/*********** METHODS ********************************************************/

template <typename T>
void MatrixV1<T>::printValues() {
    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            std::cout << this->operator[](i)[j] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
int MatrixV1<T>::getNumOfRows() {
    return this->numOfRows;
}

template <typename T>
int MatrixV1<T>::getNumOfCols() {
    return this->numOfCols;
}



#endif //PV021_MATRIX_IMPL_MatrixV1_H
