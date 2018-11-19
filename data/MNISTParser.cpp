//
// Created by Martin Galajda on 28/10/2018.
//

#include "MNISTParser.h"
#include <map>

void CSVRow::readNextRow(std::istream& str)
{
    std::string         line;
    std::getline(str, line);

    std::stringstream   lineStream(line);
    std::string         cell;

    rowCells.clear();
    while(std::getline(lineStream, cell, ','))
    {
        rowCells.push_back(cell);
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty())
    {
        // If there was a trailing comma then add an empty element.
        rowCells.push_back("");
    }
}

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
}


std::map<int, std::shared_ptr<Matrix<double>>> createOneHotEncodedMap(int start, int numOfValues) {
    std::map<int, std::shared_ptr<Matrix<double>>> oneHotEncodeMap;

    for (int i = start; i < numOfValues; i++) {
        auto newVector = std::shared_ptr<Matrix<double>>(new Matrix<double>(numOfValues, 1));

        oneHotEncodeMap[i] = newVector;
        for (int j = start; j < numOfValues; j++) {
            if (i == j) {
                (*newVector)[j][0] = 1.0;
            } else {
                (*newVector)[j][0] = 0.0;
            }
        }
    }

    return oneHotEncodeMap;
}


std::vector<CSVRow> MNISTParser::parse() {
    std::ifstream file(filename);

    std::vector<CSVRow> rows;
    rows.reserve(65000);

    CSVRow row;
    while(file >> row)
    {
        rows.push_back(row);
    }

    return rows;
}

std::vector<CSVRow> MNISTParser::parseLabels() {
    std::ifstream file(filenameForLabels);

    std::vector<CSVRow> rows;
    rows.reserve(65000);

    CSVRow row;
    while(file >> row)
    {
        rows.push_back(row);
    }

    return rows;
}

std::vector<std::shared_ptr<Matrix<double>>> MNISTParser::parseToMatrices() {
    std::vector<CSVRow> rows = this->parse();
    std::vector<std::shared_ptr<Matrix<double>>> allTrainingInstances;
    allTrainingInstances.reserve(rows.size());

    for (auto i = 0; i < rows.size(); i++) {
        CSVRow &row = rows[i];

        auto newInstanceMatrix = std::shared_ptr<Matrix<double>> (new Matrix<double>(row.size(), 1));

        for (auto j = 0; j < row.size(); j++) {
            (*newInstanceMatrix)[j][0] = std::stod(row[j]) / 255.0;
        }

        allTrainingInstances.push_back(newInstanceMatrix);
    }

    return allTrainingInstances;
}

std::vector<std::shared_ptr<Matrix<double>>> MNISTParser::parseLabelsToOneHotEncodedVectors() {
    std::vector<CSVRow> rows = this->parseLabels();

    auto oneHotEncodeMap = createOneHotEncodedMap(0, 10);

    std::vector<std::shared_ptr<Matrix<double>>> allOneHotEncodedLabels;
    allOneHotEncodedLabels.reserve(rows.size());

    for (auto i = 0; i < rows.size(); i++) {
        CSVRow &row = rows[i];

        int value = std::stoi(row[0]);

        auto oneHotEncodeVector = oneHotEncodeMap[value];
        allOneHotEncodedLabels.push_back(oneHotEncodeVector);
    }

    return allOneHotEncodedLabels;
}