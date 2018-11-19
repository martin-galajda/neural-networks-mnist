//
// Created by Martin Galajda on 28/10/2018.
//
#define _GLIBCXX_USE_CXX11_ABI 0

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include "../matrix_impl/Matrix.hpp"

#ifndef MATRIXBENCHMARKS_MNISTPARSER_H
#define MATRIXBENCHMARKS_MNISTPARSER_H

class CSVRow
{
public:
    std::string const& operator[](std::size_t index) const
    {
        return rowCells[index];
    }

    std::size_t size() const
    {
        return rowCells.size();
    }

    void readNextRow(std::istream& str);
private:
    std::vector<std::string> rowCells;
};


class MNISTParser {
public:
    MNISTParser(const std::string &filename, const std::string &filenameForLabels): filename(filename), filenameForLabels(filenameForLabels) {}

    std::vector<CSVRow> parse();
    std::vector<CSVRow> parseLabels();

    std::vector<std::shared_ptr<Matrix<double>>> parseToMatrices();
    std::vector<std::shared_ptr<Matrix<double>>> parseLabelsToOneHotEncodedVectors();

protected:
    std::string filename;
    std::string filenameForLabels;
};


#endif //MATRIXBENCHMARKS_MNISTPARSER_H
