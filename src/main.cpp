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
  // double learningRateETA = 0.25;
  // Perceptron pcn(trainInputs, trainTargets);
  // pcn.pcntrain(trainInputs, trainTargets, learningRateETA, 100);

  // pcn.confmat(testInputs, testTargets);

  // Comment out traing data functions as require.
  //  trainOr();
  // trainXOr();

  MatrixXd test(4, 10); // nData, data points
  test << -3, -1, -2, 0, 40, -3, 5, 6, 2, 0,
      2, -3, 5, 6, -2, 20, -3, 5, 6, 1,
      2, -3, 5, 60, 0, 2, -3, 5, 6, -1,
      2, -3, 5, 6, -2, 2, -3, 50, 6, 1;

  MatrixXd test3(10, 4); // nData, data points
  test3 << -3, -1, -2, 0,
      40, -3, 5, 6,
      2, 0, 2, -3,
      5, 6, -2, 20,
      -3, 5, 6, 1,
      2, -3, 5, 60,
      0, 2, -3, 5,
      6, -1, 2, -3,
      5, 6, -2, 2,
      -3, 50, 6, 1;

  MatrixXd a(4, 10);
  a.fill(1.0);
  MatrixXd b(4, 10);
  b.fill(0);

  auto test2 = (test.array() > 0).select(a, b);
  std::cout << test << std::endl;
  std::cout << test3 << std::endl;

  std::cout << "Test size: " << test.innerSize() << " " << test.outerSize() << " " << test.size() << std::endl;
  std::cout << "Test3 size: " << test.innerSize() << " " << test.outerSize() << std::endl;

  // An argmax funnction to work on one axis (the rows)
  // it should return an array of index associated with the row (ndata)
  int nData = 4;
  int dataLength = 10;
  auto arr = test.array();

  ArrayXd testAsArrMax(nData);
  int countN = 0;
  int countD = 1;
  double minDouble = std::numeric_limits<double>::min();
  std::cout << "Min Double " << minDouble << std::endl;
  double max = arr(0);
  int maxI = -1;
  std::cout << std::endl << arr << std::endl;
  std::cout << std::endl << "Seq" << std::endl;

  // std::cout << test(seqN(0, 1), seqN(0, 10)) << std::endl;
  // std::cout << test(seqN(1, 1), seqN(0, 10)) << std::endl;
  // std::cout << test(seqN(2, 1), seqN(0, 10)) << std::endl;
  // std::cout << test(seqN(3, 1), seqN(0, 10)) << std::endl;

  for (int countN = 0; countN < nData; countN++)
  {
    auto axis = test(seqN(countN, 1), seqN(0, dataLength));
    std::cout << axis << std::endl;
    // if (max < arr(i))
    // {
    //   max = arr(i);
    //   maxI = countD;
    // }
    // countD++;
    // if (countD == dataLength)
    // {
    //   testAsArrMax[countN] = maxI;
    //   countD = 1;
    //   countN++;
    //   if (countN < nData)
    //   {
    //     max = arr((countN * dataLength));
    //   }
    // }
  }

  //std::cout << testAsArrMax << std::endl;

  return EXIT_SUCCESS;
}
