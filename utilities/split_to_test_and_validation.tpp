//
// Created by Martin Galajda on 29/10/2018.
//

#include <vector>
#include <map>
#include <numeric>
#include <random>



#ifndef UTILITIES_SPLIT_TO_TEST_AND_VALIDATION
#define UTILITIES_SPLIT_TO_TEST_AND_VALIDATION
std::random_device randomDevice;     // only used once to initialise (seed) engine

template <typename T>
std::map<std::string, std::vector<int>> splitToTestAndValidationSetIndices(std::vector<T> &instanceSet,
                                                                           const double validationSize = 0.2) {

    auto allExamplesCount = instanceSet.size();

    std::vector<int> validation_indices;
    std::vector<int> training_indices(allExamplesCount);
    std::iota(training_indices.begin(), training_indices.end(), 0);

    std::mt19937 rng(randomDevice());    // random-number engine used (Mersenne-Twister in this case)

    const int VALIDATION_SET_SIZE = allExamplesCount * validationSize;

    validation_indices.reserve(VALIDATION_SET_SIZE);
    for (auto i = 0; i < VALIDATION_SET_SIZE; i++) {
        std::uniform_int_distribution<int> pickRandomInstanceFromAll(0, training_indices.size() -1); // guaranteed unbiased
        auto instanceIdx = pickRandomInstanceFromAll(randomDevice);

        validation_indices.push_back(training_indices[instanceIdx]);
        training_indices.erase(training_indices.begin() + instanceIdx);
    }

    return {{ "training", training_indices }, { "validation", validation_indices }};
}

#endif //UTILITIES_SPLIT_TO_TEST_AND_VALIDATION
