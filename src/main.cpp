//============================================================================
// Graph Traversing
//============================================================================
#include <iostream>
#include "Perceptron.h"
#include <Eigen/Dense>

using namespace ML;
using namespace Eigen;

int main(int argc, char *argv[])
{
  double learningRateETA = 0.25;

  Matrix<double, 4, 2> trainInputs;
  trainInputs << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0;

  Matrix<double, 4, 1> trainTargets;
  trainTargets << 0.0, 1.0, 1.0, 1.0;

  Perceptron pcn(trainInputs, trainTargets);
  pcn.pcntrain(trainInputs, trainTargets, learningRateETA, 6);

  return EXIT_SUCCESS;
}
