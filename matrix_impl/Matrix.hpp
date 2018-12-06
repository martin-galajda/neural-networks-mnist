// Created by Martin Galajda on 20/10/2018.
//

#include <iostream>
#include <cmath>
#include "../initializers/BaseInitializer.h"
#include "../initializers/NormalInitializer.h"
#include <stdexcept>
#include <memory>

#ifndef PV021_MATRIX_IMPL_Matrix_H
#define PV021_MATRIX_IMPL_Matrix_H

template <typename T>
class Matrix {
public:

    Matrix(int rows, int cols, int depth = 1, int batchSize = 1);

    Matrix(int rows, int cols, BaseInitializer *initializer);
    Matrix(int rows, int cols, int depth, int batchSize, BaseInitializer *initializer);

    Matrix(T* array, int rows, int cols, bool copyData = true, int depth = 1, int batchSize = 1);

  ~Matrix();

    Matrix & operator=(const Matrix&) = delete;
    Matrix(const Matrix&) = delete;


    inline T *operator[] (const int& index);
    inline T *operator() (const int& i, const int& j, const int& z);
    inline T *operator() (const int& i, const int& j, const int& z, const int& s);

    Matrix* operator*(Matrix&);
    void operator-=(Matrix &);
    void operator+=(Matrix &);
    Matrix* operator/(Matrix &);

    void operator/=(const T &);
    void operator*=(const T &);
    void operator+=(T &);

    void copyElementsFrom(Matrix<T> &other, bool copyMemory = true);

    Matrix* operator-(Matrix &);
    Matrix* operator*(const T &);

    Matrix* copy();

    Matrix* transposeToNew();
    Matrix* softmax();
    Matrix* softmaxInPlace();
    Matrix* relu();
    Matrix* reluInPlace();
    Matrix* argMaxByRow();
    Matrix* sqrt();
    Matrix* pow(double);
    Matrix* componentWiseReluDerivMult(Matrix *, bool inPlace = true);
    Matrix* componentWiseMult(Matrix *, bool inPlace = true);

    T totalAbsDifference(Matrix<T> *other);
    T absoluteSum();

    Matrix* reshape(int rows, int cols, int depth, int batchSize);

    T crossEntropyLoss(Matrix *);

    inline int getNumOfRows() const;
    inline int getNumOfCols() const;
    inline int getDepth() const { return this->depth; }
    inline int getBatchSize() const { return this->batchSize; }
    void printValues();
    void setAllElementsZero();
    void setAllElementsTo(const T &);
    std::string toString();
    void updateDebug();
protected:
    T *rawData;

    int numOfRows;
    int numOfCols;
    int depth = 1;
    int batchSize = 1;

    std::string debugValues;
    std::vector<T> values;

    bool notOwningData = false;
};


using MatrixDouble = Matrix<double>;
using MatrixDoubleSharedPtr = std::shared_ptr<Matrix<double >>;

#endif //PV021_MATRIX_IMPL_Matrix_H
