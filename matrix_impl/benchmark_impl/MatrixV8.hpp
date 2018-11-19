// Created by Martin Galajda on 20/10/2018.
//

#include <iostream>

#ifndef PV021_MATRIX_IMPL_MatrixV8_H
#define PV021_MATRIX_IMPL_MatrixV8_H

template <typename T>
class MatrixV8 {
public:
    MatrixV8(int rows, int cols);
    MatrixV8(T* array, int rows, int cols);
    MatrixV8(T* array, int rows, int cols, bool copyData);
    ~MatrixV8();

    T *operator[](const int& index);
    MatrixV8* operator*(MatrixV8&);

    MatrixV8* transposeToNew();

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
MatrixV8<T>::MatrixV8(int rows, int cols) {
    this->numOfRows = rows;
    this->numOfCols = cols;
    this->rawData = (T*) malloc(sizeof(T) * rows * cols);
}

template <typename T>
MatrixV8<T>::MatrixV8(T *data, int rows, int cols): MatrixV8(rows, cols) {
    this->numOfRows = rows;
    this->numOfCols = cols;

    for (auto i = 0; i < numOfRows * numOfCols; i++) {
        *(this->rawData + i) = *(data + i);
    }
}

template <typename T>
MatrixV8<T>::MatrixV8(T *data, int rows, int cols, bool copyData) {
    this->numOfRows = rows;
    this->numOfCols = cols;

    this->data = data;
}


/*********** DESTRUCTORS ********************************************************/

template <typename T>
MatrixV8<T>::~MatrixV8() {
    free(rawData);
}

/*********** OPERATORS ********************************************************/



template <typename T>
T* MatrixV8<T>::operator[](const int& index) {
    return this->rawData + (index * numOfCols);
}

template <typename T>
MatrixV8<T>* MatrixV8<T>::operator*(MatrixV8& other) {
    if (this->getNumOfCols() != other.getNumOfRows()) {
        throw std::invalid_argument("Matrices with incompatible dimension passed for multiplication.");
    }

    auto thisNumOfRows = this->getNumOfRows();
    auto thisNumOfCols = this->getNumOfCols();
    auto otherNumOfCols = other.getNumOfCols();

    auto resultMatrix = new MatrixV8(thisNumOfRows, otherNumOfCols);

    // A = (m,n)
    // B = (n,j)
    // C = A * B
    // C = (m,j)
    //
    //      A        *      B        =                                  C
    // |a11 ... a1n| * |b11 ... b1j| = | a11 * b11 + ... + a1n * bn1 | ... | a11 * b1j + ... + a1n * bnj |
    // [... ... ...|   [... ... ...|   |             ...             | ... |             ...             |
    // |am1 ... amn|   |bn1 ... bnj|   | am1 * b11 + ... + amn * bn1 | ... | am1 * b1j + ... + amn * bnj |


    const TILE_SIZE = 13;

    for (auto i = 0; i < thisNumOfRows; i += TILE_SIZE) {
        const int maxI = std::min(i + TILE_SIZE, thisNumOfRows);

        for (auto j = 0; j < otherNumOfCols; j+= TILE_SIZE) {
            const int maxJ = std::min(j + TILE_SIZE, otherNumOfCols);

            T sum = 0.0;
            for (auto k = 0; k < thisNumOfCols; k+= TILE_SIZE) {
                const int maxK = std::min(k + TILE_SIZE, thisNumOfCols);
                for (auto i2 = i; i2 < maxI; i2++) {
                    for (auto j2 = j; j2 < maxJ; j2++) {
                        for (auto k2 = k; k2 < maxK; k2++) {
                            sum += (*this)[i2][k2] * other[k2][j2];
                        }

                        resultMatrix[i2][j2] += sum;
                    }
                }
            }
        }
    }

    return resultMatrix;
}

/*********** METHODS ********************************************************/

template <typename T>
void MatrixV8<T>::printValues() {
    for (auto i = 0; i < numOfRows; i++) {
        for (auto j = 0; j < numOfCols; j++) {
            std::cout << this->operator[](i)[j] << " ";
        }
        std::cout << std::endl;
    }
}


template <typename T>
int MatrixV8<T>::getNumOfRows() {
    return this->numOfRows;
}

template <typename T>
int MatrixV8<T>::getNumOfCols() {
    return this->numOfCols;
}

template <typename T>
std::string& MatrixV8<T>::toString(){
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
MatrixV8<T>* MatrixV8<T>::transposeToNew(){
    T *newArrayData = (T *) malloc (sizeof(T) * this->getNumOfRows() * this->getNumOfCols());

    for (auto i = 0; i < this->getNumOfRows(); i++) {
        for (auto j = 0; j < this->getNumOfCols(); j++) {
            auto value = (*this)[i][j];
            newArrayData[(j * this->getNumOfCols()) + i] = value;
        }
    }

    return new MatrixV8<T>(newArrayData, this->getNumOfCols(), this->getNumOfRows(), false);
}

template <typename T>
void MatrixV8<T>::transpose(){
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
void MatrixV8<T>::setElements(T *newElements){
    free(this->rawData);

    this->rawData = newElements;
}


#endif //PV021_MATRIX_IMPL_MatrixV8_H
