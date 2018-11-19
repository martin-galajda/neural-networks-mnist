// Created by Martin Galajda on 20/10/2018.
//
#define _GLIBCXX_USE_CXX11_ABI 0

#include <iostream>
#include <cmath>
#include "../initializers/BaseInitializer.h"
#include "../initializers/NormalInitializer.h"
#include <stdexcept>

#ifndef PV021_MATRIX_IMPL_Matrix_H
#define PV021_MATRIX_IMPL_Matrix_H

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
class Matrix {
public:

    Matrix(int rows, int cols);
    Matrix(int rows, int cols, BaseInitializer *initializer);

    Matrix(T* array, int rows, int cols);
    Matrix(T* array, int rows, int cols, bool copyData);
    ~Matrix();

    Matrix & operator=(const Matrix&) = delete;
    Matrix(const Matrix&) = delete;


    inline T *operator[] (const int& index);

    Matrix* operator*(Matrix&);
    void operator-=(Matrix &);
    void operator+=(Matrix &);
    Matrix* operator/(Matrix &);

    void operator/=(const T &);
    void operator*=(const T &);
    void operator+=(T &);

    void copyElementsFrom(Matrix<T> &other);

    Matrix* operator-(Matrix &);
    Matrix* operator*(const T &);

    Matrix* transposeToNew();
    Matrix* softmax();
    Matrix* softmaxInPlace();
    Matrix* relu();
    Matrix* reluInPlace();
    Matrix* argMaxByRow();
    Matrix* sqrt();
    Matrix* pow(double);

    T crossEntropyLoss(Matrix *);

    inline int getNumOfRows() const;
    inline int getNumOfCols() const;
    void printValues();
    void setAllElementsZero(const T &value);
    std::string toString();
    void updateDebug();


protected:
    RowVector<T>** rows;
    T *rawData;

    int numOfRows;
    int numOfCols;

    std::string debugValues;
    std::vector<T> values;
};


#endif //PV021_MATRIX_IMPL_Matrix_H
