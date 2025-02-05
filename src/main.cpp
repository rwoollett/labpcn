//============================================================================
// Graph Traversing
//============================================================================
#include <iostream>
#include "Perceptron.h"
#include "TrainingData.h"
#include <numeric>
#include <Eigen/Dense>
#include <algorithm>


using namespace ML;
using namespace Eigen;
using namespace ML::DataSet;

int main(int argc, char *argv[])
{
  // Todo read the pima datafile into the data for pcn
  // trainset, targetset,
  MatrixXd dataSet = readDataFile("../dataset/pima-indians-diabetes.data");
  std::cout << dataSet.innerSize() << " " << dataSet.outerSize() << std::endl;

  MatrixXd trainTargets = dataSet(seqN(1, dataSet.innerSize() / 2, 2), last);
  MatrixXd testTargets = dataSet(seqN(0, dataSet.innerSize() / 2, 2), last);
  std::cout << trainTargets.innerSize() << " " << trainTargets.outerSize() << std::endl;
  std::cout << testTargets.innerSize() << " " << testTargets.outerSize() << std::endl;

  // std::cout << trainTargets << std::endl;
  // std::cout << testTargets << std::endl;

  MatrixXd trainInputs = dataSet(seqN(1, dataSet.innerSize() / 2, 2), seqN(0, dataSet.outerSize() - 1));
  MatrixXd testInputs = dataSet(seqN(0, dataSet.innerSize() / 2, 2), seqN(0, dataSet.outerSize() - 1));
  std::cout << trainInputs.innerSize() << " " << trainInputs.outerSize() << std::endl;
  std::cout << testInputs.innerSize() << " " << testInputs.outerSize() << std::endl;
  // std::cout << trainInputs << std::endl;

  // //  # Perceptron training on the original pima dataset
  double learningRateETA = 0.25;
  Perceptron pcn(trainInputs, trainTargets);
  pcn.pcntrain(trainInputs, trainTargets, learningRateETA, 100);

  pcn.confmat(testInputs, testTargets);

  // Comment out traing data functions as require.
    trainOr();
  // trainXOr();

  // MatrixXd test(4, 10); // nData, data points
  // test << -3, -1, -2, 0, 40, -3, 5, 6, 2, 0,
  //     2, -3, 5, 6, -2, 20, -3, 5, 6, 1,
  //     60, -3, 5, 60, 0, 2, -3, 5, 6, -1,
  //     2, -3, 5, 6, -2, 2, -3, 50, 6, 1;

  // std::cout << "Test size: " << test.innerSize() << " " << test.outerSize() << " " << test.size() << std::endl;

  // // An argmax funnction to work on one axis (the rows)
  // // it should return an array of index associated with the row (ndata)
  // int nData = 4;
  // int dataLength = 10;

  // ArrayXd testAsArrMax = pcn.indiceMax(test, nData, dataLength);
  // std::cout << "TestAsIndiceMax: " << std::endl << testAsArrMax << std::endl;

  return EXIT_SUCCESS;
}
