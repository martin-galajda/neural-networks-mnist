// Created by Martin Galajda on 20/10/2018.
//

#include <iostream>

#ifndef PV021_MATRIX_IMPL_MatrixV7_H
#define PV021_MATRIX_IMPL_MatrixV7_H

template <typename T>
class MatrixV7 {
public:
    MatrixV7(int rows, int cols);
    MatrixV7(T* array, int rows, int cols);
    MatrixV7(T* array, int rows, int cols, bool copyData);
    ~MatrixV7();

    T *operator[](const int& index);
    MatrixV7* operator*(MatrixV7&);

    MatrixV7* transposeToNew();

    int getNumOfRows();
    int getNumOfCols();
    void printValues();
    std::string &toString();
    void transpose();
    void setElements(T*);

protected:
    T *rawData;

    int numOfRows;
    int numOfCols;
};



/*********** CONSTRUCTORS ********************************************************/

template <typename T>
MatrixV7<T>::MatrixV7(int rows, int cols) {
    this->numOfRows = rows;
    this->numOfCols = cols;
    this->rawData = (T*) malloc(sizeof(T) * rows * cols);
}

template <typename T>
MatrixV7<T>::MatrixV7(T *data, int rows, int cols): MatrixV7(rows, cols) {
    this->numOfRows = rows;
    this->numOfCols = cols;

    for (auto i = 0; i < numOfRows * numOfCols; i++) {
        *(this->rawData + i) = *(data + i);
    }
}

template <typename T>
MatrixV7<T>::MatrixV7(T *data, int rows, int cols, bool copyData) {
    this->numOfRows = rows;
    this->numOfCols = cols;

    this->data = data;
}


/*********** DESTRUCTORS ********************************************************/

template <typename T>
MatrixV7<T>::~MatrixV7() {
    free(rawData);
}

/*********** OPERATORS ********************************************************/



template <typename T>
T* MatrixV7<T>::operator[](const int& index) {
    return this->rawData + (index * numOfCols);
}

template <typename T>
MatrixV7<T>* MatrixV7<T>::operator*(MatrixV7& other) {
    if (this->getNumOfCols() != other.getNumOfRows()) {
        throw std::invalid_argument("Matrices with incompatible dimension passed for multiplication.");
    }

    auto thisNumOfRows = this->getNumOfRows();
    auto thisNumOfCols = this->getNumOfCols();
    auto otherNumOfCols = other.getNumOfCols();

    auto resultMatrix = new MatrixV7(thisNumOfRows, otherNumOfCols);

    // A = (m,n)
    // B = (n,j)
    // C = A * B
    // C = (m,j)
    //
    //      A        *      B        =                                  C
    // |a11 ... a1n| * |b11 ... b1j| = | a11 * b11 + ... + a1n * bn1 | ... | a11 * b1j + ... + a1n * bnj |
    // [... ... ...|   [... ... ...|   |             ...             | ... |             ...             |
    // |am1 ... amn|   |bn1 ... bnj|   | am1 * b11 + ... + amn * bn1 | ... | am1 * b1j + ... + amn * bnj |

    auto *otherTransposed = other.transpose();

    // Iterate over every row from Matrix A
    for (auto resultRow = 0; resultRow < thisNumOfRows; resultRow++) {
        // Iterate over every column from Matrix B
        for (auto resultCol = 0; resultCol < otherNumOfCols; resultCol++) {
            T result = 0;
            // Perform sum of the dot product between two matching vectors from A and B
            for (auto sumIndex = 0; sumIndex < thisNumOfCols; sumIndex++) {
                // From A we take a row vector and from B we take a column vector
                result += (*this)[resultRow][sumIndex] * (*otherTransposed)[resultCol][sumIndex];
            }

            (*resultMatrix)[resultRow][resultCol] = result;
        }
    }

    free(otherTransposed);

    return resultMatrix;
}

/*********** METHODS ********************************************************/

template <typename T>
void MatrixV7<T>::printValues() {
    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            std::cout << this->operator[](i)[j] << " ";
        }
        std::cout << std::endl;
    }
}


template <typename T>
int MatrixV7<T>::getNumOfRows() {
    return this->numOfRows;
}

template <typename T>
int MatrixV7<T>::getNumOfCols() {
    return this->numOfCols;
}

template <typename T>
std::string& MatrixV7<T>::toString(){
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
MatrixV7<T>* MatrixV7<T>::transposeToNew(){
    T *newArrayData = (T *) malloc (sizeof(T) * this->getNumOfRows() * this->getNumOfCols());

    for (auto i = 0; i < this->getNumOfRows(); i++) {
        for (auto j = 0; j < this->getNumOfCols(); j++) {
            auto value = (*this)[i][j];
            newArrayData[(j * this->getNumOfCols()) + i] = value;
        }
    }

    return new MatrixV7<T>(newArrayData, this->getNumOfCols(), this->getNumOfRows(), false);
}

template <typename T>
void MatrixV7<T>::transpose(){
    T *newArrayData = (T *) malloc (sizeof(T) * this->getNumOfRows() * this->getNumOfCols());

    for (auto i = 0; i < this->getNumOfRows(); i++) {
        for (auto j = 0; j < this->getNumOfCols(); j++) {
            auto value = (*this)[i][j];
            newArrayData[(j * this->getNumOfCols()) + i] = value;
        }
    }

    this->setElements(newArrayData);
}

template <typename T>
void MatrixV7<T>::setElements(T *newElements){
    free(this->rawData);

    this->rawData = newElements;
}


#endif //PV021_MATRIX_IMPL_MatrixV7_H
