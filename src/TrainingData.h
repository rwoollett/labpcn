#pragma once

#include "Perceptron.h"
#include <Eigen/Dense>
#include <iostream>

using namespace ML;
using namespace Eigen;

namespace ML::DataSet
{
  void trainOr();

  void trainXOr();

  MatrixXd readDataFile(std::string fileName);

  std::vector<double> splitStringToDouble(const std::string &str, char delimiter);

  std::string stripFileLine(std::string line);

  bool isCommentLine(std::string line);

  std::tuple<int, int> readDataShapeFromFile(std::string fileName);

}