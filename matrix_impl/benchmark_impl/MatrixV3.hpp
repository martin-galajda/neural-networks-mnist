//
// Created by Martin Galajda on 20/10/2018.
//

#ifndef PV021_MATRIX_IMPL_MatrixV3_H
#define PV021_MATRIX_IMPL_MatrixV3_H


#include <vector>
#include <iostream>

template <typename T>
class MatrixV3 {
public:
    MatrixV3(int numOfRows, int numOfCols) {
        this->rows = std::vector<std::vector<T> > (numOfRows, std::vector<T> (numOfCols));
    }
    MatrixV3(T *data, int numOfRows, int numOfCols): MatrixV3(numOfRows, numOfCols) {
        for (auto rowOffset = 0; rowOffset < numOfRows; rowOffset++) {
            for (auto colOffset = 0; colOffset < numOfCols; colOffset++) {
                T arrayValue = *(data + (rowOffset * numOfCols) + colOffset);
                (*this)[rowOffset][colOffset] = arrayValue;
            }
        }
    }
//
    int getNumOfRows() {
        return this->rows.size();
    }
    int getNumOfCols() {
        return this->rows.at(0).size();
    }
//
    std::vector<T> &operator[](const int& index) {
        return this->rows.at(index);
    }

    void printValues() {
        for (auto rowOffset = 0; rowOffset < getNumOfRows(); rowOffset++) {
            for (auto colOffset = 0; colOffset < getNumOfCols(); colOffset++) {
                std::cout << (*this)[rowOffset][colOffset] << " ";
            }
            std::cout << std::endl;
        }
    }


    MatrixV3<T>* operator*(MatrixV3 &other) {
        if (this->getNumOfCols() != other.getNumOfRows()) {
            throw std::invalid_argument("Matrices with incompatible dimension passed for multiplication.");
        }

        auto thisNumOfRows = this->getNumOfRows();
        auto thisNumOfCols = this->getNumOfCols();
        auto otherNumOfCols = other.getNumOfCols();

        auto resultMatrix = new MatrixV3(thisNumOfRows, otherNumOfCols);

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

private:
    std::vector<std::vector<T> > rows;
};


#endif //PV021_MATRIX_IMPL_MatrixV3_H
