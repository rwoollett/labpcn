//============================================================================
// Graph Traversing
//============================================================================
#include <iostream>
#include "Perceptron.h"
#include "TrainingData.h"
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
  std::cout << dataSet;


  // Comment out traing data functions as require.
  // trainOr();
  // trainXOr();


  return EXIT_SUCCESS;
}
