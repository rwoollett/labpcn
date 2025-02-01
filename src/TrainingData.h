#pragma once

#include "Perceptron.h"
#include <Eigen/Dense>

using namespace ML;
using namespace Eigen;

namespace ML::DataSet
{
  void trainOr()
  {
    double learningRateETA = 0.25;

    MatrixXd trainInputs(4, 2);
    trainInputs << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0;

    MatrixXd trainTargets(4, 1);
    trainTargets << 0.0, 1.0, 1.0, 1.0;

    Perceptron pcn(trainInputs, trainTargets);
    pcn.pcntrain(trainInputs, trainTargets, learningRateETA, 6);
  }

  void trainXOr()
  {
    double learningRateETA = 0.25;

    MatrixXd trainInputs(4, 3);
    trainInputs << 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0;
    // MatrixXd trainInputs(4, 2);
    // trainInputs << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0;

    MatrixXd trainTargets(4, 1);
    trainTargets << 0.0, 1.0, 1.0, 0.0;

    Perceptron pcn(trainInputs, trainTargets);
    pcn.pcntrain(trainInputs, trainTargets, learningRateETA, 14);
  }

}