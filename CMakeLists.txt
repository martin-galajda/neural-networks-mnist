cmake_minimum_required(VERSION 2.8.12.2)

project(pv021_neural_networks_project)
project(cnn)
project(projectTests)
project(matrixBenchmarks)

set(CMAKE_CXX_STANDARD 17)

set(NN_SOURCES
        matrix_impl/Matrix.hpp
        matrix_impl/Matrix.cpp

        initializers/BaseInitializer.cpp
        initializers/BaseInitializer.h
        initializers/ZeroInitializer.cpp
        initializers/ZeroInitializer.h
        initializers/NormalInitializer.cpp
        initializers/NormalInitializer.h
        initializers/XavierInitializer.cpp
        initializers/XavierInitializer.h

        layers/BaseLayer.cpp
        layers/BaseLayer.h
        layers/DenseLayer.cpp
        layers/DenseLayer.h
        layers/ActivationFunction.h
        layers/L2Regularizer.cpp
        layers/L2Regularizer.h

        neural_network/ComputationalGraph.cpp
        neural_network/ComputationalGraph.h

        data/MNISTParser.cpp
        data/MNISTParser.h

        utilities/split_to_test_and_validation.tpp
        utilities/populate_placeholders.h
        utilities/populate_placeholders.cpp
        utilities/config_params_utils.h
        utilities/config_params_utils.cpp
        utilities/report_accuracy.h
        utilities/report_accuracy.cpp


        neural_network/MiniBatchOptimizer.cpp
        neural_network/MiniBatchOptimizer.h
        neural_network/BaseOptimizer.cpp
        neural_network/BaseOptimizer.h
        neural_network/MomentumOptimizer.cpp
        neural_network/MomentumOptimizer.h
        neural_network/AdamOptimizer.cpp
        neural_network/AdamOptimizer.h

        layers/ConvLayer.cpp
        layers/ConvLayer.h
        ops/Convolution.cpp
        ops/Convolution.h
        layers/FlattenLayer.cpp
        layers/FlattenLayer.h
        layers/MaxPool2DLayer.cpp
        layers/MaxPool2DLayer.h
        layers/BatchNormalization.cpp layers/BatchNormalization.h neural_network/GraphBuilder.cpp neural_network/GraphBuilder.h)

set(SOURCES
        tests/matrix_tests.cpp
        tests/xor_problem_tests.cpp
        tests/dense_layer_tests.cpp
        tests/mnist_parser_tests.cpp
#        tests/mnist_problem_tests.cpp
        tests/ops_tests.cpp
        tests/conv_layer_tests.cpp
        tests/forward_pass_layers_test.cpp

        ${NN_SOURCES} tests/max_pool_layer_tests.cpp utilities/test_utils.cpp utilities/test_utils.h )

include_directories(./googletest-1.8.1/googletest/include ./googletest-1.8.1/googletest ./tests ./benchmarks ./matrix_impl ./initializers ./layers ./data)

add_executable(pv021_neural_networks_project main.cpp ${NN_SOURCES})
add_executable(projectTests ${SOURCES})
add_executable(matrixBenchmarks benchmarks/matrix_op_benchmarks.cpp matrix_impl/benchmark_impl/MatrixV1.hpp matrix_impl/benchmark_impl/MatrixV3.hpp matrix_impl/benchmark_impl/MatrixV4.hpp matrix_impl/benchmark_impl/MatrixV5.hpp matrix_impl/benchmark_impl/MatrixV6.hpp matrix_impl/benchmark_impl/MatrixV7.hpp layers/DenseLayer.cpp layers/DenseLayer.h initializers/ZeroInitializer.cpp initializers/ZeroInitializer.h initializers/BaseInitializer.cpp initializers/BaseInitializer.h layers/ActivationFunction.h)
add_executable(cnn ${NN_SOURCES} main3.cpp)

add_library(gtest ./googletest-1.8.1/googletest/src/gtest-all.cc)
add_library(gtest_main ./googletest-1.8.1/googletest/src/gtest_main.cc)

target_link_libraries(projectTests gtest gtest_main)

target_compile_options(pv021_neural_networks_project PUBLIC -O3 -Wall)
target_compile_options(cnn PUBLIC -O3 -Wall)


enable_testing()