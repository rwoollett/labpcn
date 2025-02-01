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

  MatrixXd trainInputs(4, 2);
  trainInputs << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0;

  MatrixXd trainTargets(4, 1);
  trainTargets << 0.0, 1.0, 1.0, 1.0;

  Perceptron pcn(trainInputs, trainTargets);
  pcn.pcntrain(trainInputs, trainTargets, learningRateETA, 6);

  return EXIT_SUCCESS;
}
