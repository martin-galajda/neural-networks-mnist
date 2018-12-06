//
// Created by Martin Galajda on 02/12/2018.
//


#include "gtest/gtest.h"
#include "../ops/Convolution.h"
#include "../layers/ConvLayer.h"
#include "../layers/DenseLayer.h"
#include "../layers/FlattenLayer.h"
#include "../initializers/ZeroInitializer.h"
#include "../neural_network/ComputationalGraph.h"
#include "../utilities/test_utils.h"

namespace LayersIntegrationTestsData {
  MatrixDoubleSharedPtr getInputs() {
    double result[] = {
      -0., 1., 2., 3., 4., -5., 6.,
      7., 8., 9., -10., 11., 12., 13.,
      14., -15., 16., 17., 18., 19., -20.,
      21., 22., 23., 24., -25., 26., 27.,
      28., 29., -30., 31., 32., 33., 34., -35., 36., 37., 38., 39.,
      -40., 41.,42., 43., 44., -45., 46., 47., 48.,

      49.,-50.,51.,52.,53.,54.,-55.,56.,57.,58.,59.,-60.,61.,62.,
      63.,64.,-65.,66.,67.,68.,69.,-70.,71.,72.,73.,74.,-75.,76.,
      77.,78.,79.,-80.,81.,82.,83.,84.,-85.,86.,87.,88.,89.,-90.,
      91.,92.,93.,94.,-95.,96.,97.,
    };

    auto matrix = MatrixDoubleSharedPtr(new Matrix<double>(result, 7, 7, true, 1, 2));

    return matrix;
  }

  MatrixDoubleSharedPtr getUltimateResult() {
    double result[] = {
      0.00000000e+00,7.32572100e+07,0.00000000e+00,7.54639180e+07,
      0.00000000e+00,7.76706260e+07,0.00000000e+00,7.98773340e+07,
      0.00000000e+00,8.20840420e+07,0.00000000e+00,8.42907500e+07,
      0.00000000e+00,8.64974580e+07,0.00000000e+00,8.87041660e+07,
      0.00000000e+00,9.09108740e+07,0.00000000e+00,9.31175820e+07,
      0.00000000e+00,9.53242900e+07,0.00000000e+00,9.75309980e+07,
      0.00000000e+00,9.97377060e+07,0.00000000e+00,1.01944414e+08,
      0.00000000e+00,1.04151122e+08,0.00000000e+00,1.06357830e+08,
      0.00000000e+00,1.08564538e+08,0.00000000e+00,1.10771246e+08,
      0.00000000e+00,1.12977954e+08,0.00000000e+00,1.15184662e+08,
      0.00000000e+00,1.17391370e+08,0.00000000e+00,1.19598078e+08,
      0.00000000e+00,1.21804786e+08,0.00000000e+00,1.24011494e+08,
      0.00000000e+00,1.26218202e+08,0.00000000e+00,1.28424910e+08,
      0.00000000e+00,1.30631618e+08,0.00000000e+00,1.32838326e+08,
      0.00000000e+00,1.35045034e+08,0.00000000e+00,1.37251742e+08,
      0.00000000e+00,1.39458450e+08,0.00000000e+00,1.41665158e+08,
      0.00000000e+00,1.43871866e+08,0.00000000e+00,1.46078574e+08,
      0.00000000e+00,1.48285282e+08,0.00000000e+00,1.50491990e+08,
      0.00000000e+00,1.52698698e+08,0.00000000e+00,1.54905406e+08,
      0.00000000e+00,1.57112114e+08,0.00000000e+00,1.59318822e+08,
      0.00000000e+00,1.61525530e+08,0.00000000e+00,1.63732238e+08,
      0.00000000e+00,1.65938946e+08,0.00000000e+00,1.68145654e+08,
      0.00000000e+00,1.70352362e+08,0.00000000e+00,1.72559070e+08,
      0.00000000e+00,1.74765778e+08,0.00000000e+00,1.76972486e+08,
      0.00000000e+00,1.79179194e+08,0.00000000e+00,1.81385902e+08,
      0.00000000e+00,1.83592610e+08,0.00000000e+00,1.85799318e+08,
      0.00000000e+00,1.88006026e+08,0.00000000e+00,1.90212734e+08,
      0.00000000e+00,1.92419442e+08,0.00000000e+00,1.94626150e+08,
      0.00000000e+00,1.96832858e+08,0.00000000e+00,1.99039566e+08,
      0.00000000e+00,2.01246274e+08,0.00000000e+00,2.03452982e+08,
      0.00000000e+00,2.05659690e+08,0.00000000e+00,2.07866398e+08,
      0.00000000e+00,2.10073106e+08,0.00000000e+00,2.12279814e+08
    };

//  double result[] = {
//    0., 3358162., 0., 5564870.
//  };

    auto matrix = MatrixDoubleSharedPtr(new Matrix<double>(result, 128, 1, true, 1, 1));
//  auto matrix = MatrixDoubleSharedPtr(new Matrix<double>(result, 4, 1, true, 1, 1));

    return matrix;
  }

  MatrixDoubleSharedPtr getConv1Weights() {
    double weights[] = {
      0.,
      2.,
      4.,
      6.,
      8.,
      10.,
      12.,
      14.,
      16.,

      1.,
      3.,
      5.,
      7.,
      9.,
      11.,
      13.,
      15.,
      17.
    };

    auto matrix = MatrixDoubleSharedPtr(new Matrix<double>(weights, 3, 3, true, 1, 2));

    return matrix;
  }

  MatrixDoubleSharedPtr getConv1Result() {
    double result[] = {
      420.,
      824.,
      468.,

      828.,
      772.,
      1676.,

      2196.,
      1740.,
      2504.,

      // second filter ouput
      462.,
      894.,
      526.,

      936.,
      878.,
      1820.,

      2390.,
      1932.,
      2784.
    };

    auto matrix = MatrixDoubleSharedPtr(new Matrix<double>(result, 3, 3, true, 2, 1));

    return matrix;
  }

  MatrixDoubleSharedPtr getConv2Weights() {
//  double weights[] = {
//    0., 2., 4.,
//    6., 8., 10.,
//    12., 14., 16.,
//
//    18., 20., 22.,
//    24., 26., 28.,
//    30., 32., 34.,
//
//
//    1., 3., 5.,
//    7, 9, 11,
//    13, 15, 17,
//
//    19, 21, 23,
//    25, 27, 29,
//    31, 33, 35.
//  };
    double weights[] = {
      0.000000,4.000000,8.000000,
      12.000000,16.000000,20.000000,
      24.000000,28.000000,32.000000,


      2.000000,6.000000,10.000000,
      14.000000,18.000000,22.000000,
      26.000000,30.000000,34.000000,


      1.000000,5.000000,9.000000,
      13.000000,17.000000,21.000000,
      25.000000,29.000000,33.000000,


      3.000000,7.000000,11.000000,
      15.000000,19.000000,23.000000,
      27.000000,31.000000,35.000000
    };

    auto matrix = MatrixDoubleSharedPtr(new Matrix<double>(weights, 3, 3, true, 2, 2));

    return matrix;
  }

  MatrixDoubleSharedPtr getConv2Result() {
    double result[] = {
      539652., 563702.
    };

    auto matrix = MatrixDoubleSharedPtr(new Matrix<double>(result, 1, 1, true, 2, 1));

    return matrix;
  }

  MatrixDoubleSharedPtr getFlattenResult() {
    double result[] = {
      539652., 563702.
    };

    auto matrix = MatrixDoubleSharedPtr(new Matrix<double>(result, 2, 1, true, 1, 1));

    return matrix;
  }

  MatrixDoubleSharedPtr zeros(int rows,int cols, int depth, int batchSize) {
    auto matrix = MatrixDoubleSharedPtr(new Matrix<double>(rows, cols, depth, batchSize));
    matrix->setAllElementsZero();
    return matrix;
  }

  MatrixDoubleSharedPtr ones(int rows,int cols, int depth, int batchSize) {
    auto matrix = MatrixDoubleSharedPtr(new Matrix<double>(rows, cols, depth, batchSize));

    for (auto b = 0; b < matrix->getBatchSize(); b++) {
      for (auto i = 0; i < matrix->getNumOfRows(); i++) {
        for (auto j = 0; j < matrix->getNumOfCols(); j++) {
          for (auto d = 0; d < matrix->getDepth(); d++) {
            *(*matrix)(i,j,d,b) = 1.0;
          }
        }
      }
    }

    return matrix;
  }

  MatrixDoubleSharedPtr getDense1Weights() {
    double weights[] = {
      -0.000000,-128.000000,
      1.000000,129.000000,-2.000000,-130.000000,3.000000,131.000000,-4.000000,-132.000000,
      5.000000,133.000000,-6.000000,-134.000000,7.000000,135.000000,-8.000000,-136.000000,
      9.000000,137.000000,-10.000000,-138.000000,11.000000,139.000000,-12.000000,-140.000000,
      13.000000,141.000000,-14.000000,-142.000000,15.000000,143.000000,-16.000000,-144.000000,
      17.000000,145.000000,-18.000000,-146.000000,19.000000,147.000000,-20.000000,-148.000000,
      21.000000,149.000000,-22.000000,-150.000000,23.000000,151.000000,-24.000000,-152.000000,
      25.000000,153.000000,-26.000000,-154.000000,27.000000,155.000000,-28.000000,-156.000000,
      29.000000,157.000000,-30.000000,-158.000000,31.000000,159.000000,-32.000000,-160.000000,
      33.000000,161.000000,-34.000000,-162.000000,35.000000,163.000000,-36.000000,-164.000000,
      37.000000,165.000000,-38.000000,-166.000000,39.000000,167.000000,-40.000000,-168.000000,
      41.000000,169.000000,-42.000000,-170.000000,43.000000,171.000000,-44.000000,-172.000000,
      45.000000,173.000000,-46.000000,-174.000000,47.000000,175.000000,-48.000000,-176.000000,
      49.000000,177.000000,-50.000000,-178.000000,51.000000,179.000000,-52.000000,-180.000000,
      53.000000,181.000000,-54.000000,-182.000000,55.000000,183.000000,-56.000000,-184.000000,
      57.000000,185.000000,-58.000000,-186.000000,59.000000,187.000000,-60.000000,-188.000000,
      61.000000,189.000000,-62.000000,-190.000000,63.000000,191.000000,-64.000000,-192.000000,
      65.000000,193.000000,-66.000000,-194.000000,67.000000,195.000000,-68.000000,-196.000000,
      69.000000,197.000000,-70.000000,-198.000000,71.000000,199.000000,-72.000000,-200.000000,
      73.000000,201.000000,-74.000000,-202.000000,75.000000,203.000000,-76.000000,-204.000000,
      77.000000,205.000000,-78.000000,-206.000000,79.000000,207.000000,-80.000000,-208.000000,
      81.000000,209.000000,-82.000000,-210.000000,83.000000,211.000000,-84.000000,-212.000000,
      85.000000,213.000000,-86.000000,-214.000000,87.000000,215.000000,-88.000000,-216.000000,
      89.000000,217.000000,-90.000000,-218.000000,91.000000,219.000000,-92.000000,-220.000000,
      93.000000,221.000000,-94.000000,-222.000000,95.000000,223.000000,-96.000000,-224.000000,
      97.000000,225.000000,-98.000000,-226.000000,99.000000,227.000000,-100.000000,-228.000000,
      101.000000,229.000000,-102.000000,-230.000000,103.000000,231.000000,-104.000000,-232.000000,
      105.000000,233.000000,-106.000000,-234.000000,107.000000,235.000000,-108.000000,-236.000000,
      109.000000,237.000000,-110.000000,-238.000000,111.000000,239.000000,-112.000000,-240.000000,
      113.000000,241.000000,-114.000000,-242.000000,115.000000,243.000000,-116.000000,-244.000000,
      117.000000,245.000000,-118.000000,-246.000000,119.000000,247.000000,-120.000000,-248.000000,
      121.000000,249.000000,-122.000000,-250.000000,123.000000,251.000000,-124.000000,-252.000000,
      125.000000,253.000000,-126.000000,-254.000000,127.000000,255.000000
    };
//  double weights[] = {
//    -0.,    1.,   -2.,    3.,
//    -4.,    5.,   -6.,    7.
//  };
//  double weights[] = {
//    -0.,-4,
//    1, 5,
//    -2, -6,
//    3, 7
//  };

    auto matrix = MatrixDoubleSharedPtr(new Matrix<double>(weights, 128, 2, true, 1, 1));
//  auto matrix = MatrixDoubleSharedPtr(new Matrix<double>(weights, 4, 2, true, 1, 1));

    return matrix;
  }
}

TEST(layers, forwardPass)
{
  auto graph = new ComputationalGraph(7, 7, 128);
  auto zeroInitializer = new ZeroInitializer();
  auto BATCH_SIZE = 2;

  auto conv1 = new ConvLayer(3, 3, BATCH_SIZE, 1, 2, zeroInitializer, ActivationFunction::relu, 2);
  auto conv2 = new ConvLayer(3, 3, BATCH_SIZE, 2, 2, zeroInitializer, ActivationFunction::relu, 2);
  auto flatten = new FlattenLayer(BATCH_SIZE);
  auto dense1 = new DenseLayer(2, 128, BATCH_SIZE, zeroInitializer, ActivationFunction::relu);
  auto dense2 = new DenseLayer(128, 10, BATCH_SIZE, zeroInitializer, ActivationFunction::softmax);

  conv1->getWeights().copyElementsFrom(*LayersIntegrationTestsData::getConv1Weights());
  conv2->getWeights().copyElementsFrom(*LayersIntegrationTestsData::getConv2Weights());

  dense1->getWeights().copyElementsFrom(*LayersIntegrationTestsData::getDense1Weights());

  graph->addLayer(conv1);
  graph->addLayer(conv2);
  graph->addLayer(flatten);
  graph->addLayer(dense1);
  graph->addLayer(dense2);

  assertSameMatrices(LayersIntegrationTestsData::getConv1Weights(), conv1->getWeights());
  assertSameMatrices(LayersIntegrationTestsData::getConv2Weights(), conv2->getWeights());
  assertSameMatrices(LayersIntegrationTestsData::getDense1Weights(), dense1->getWeights());

  assertSameMatrices(LayersIntegrationTestsData::zeros(128, 1, 1, 1), dense1->getBiases());

  auto conv1Output = conv1->forwardPropagate(LayersIntegrationTestsData::getInputs());
  assertSameMatrices(LayersIntegrationTestsData::getConv1Result(), conv1Output);

  auto conv2Output = conv2->forwardPropagate(conv1Output);
  assertSameMatrices(LayersIntegrationTestsData::getConv2Result(), conv2Output);
  auto flattenOutput = flatten->forwardPropagate(conv2Output);
  assertSameMatrices(LayersIntegrationTestsData::getFlattenResult(), flattenOutput);

  auto denseOutput = dense1->forwardPropagate(flattenOutput);
  assertSameMatrices(LayersIntegrationTestsData::getUltimateResult(), denseOutput);
  auto output = graph->forwardPass(LayersIntegrationTestsData::getInputs());
  auto resultExpected = LayersIntegrationTestsData::getUltimateResult();
  assertSameMatrices(resultExpected, output);

//  auto lossDerivatives = ones(10, 1, 1, BATCH_SIZE);
//  *lossDerivatives /= 2;
//  auto derivativesInputs = graph->backwardPass(lossDerivatives);


}