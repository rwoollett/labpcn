#ifndef ML_PERCEPTRON_H
#define ML_PERCEPTRON_H

#include <Eigen/Dense>

using namespace Eigen;

namespace ML
{

  class Perceptron
  {
    int m_nIn;
    int m_nOut;
    int m_nData;
    MatrixXd m_weights;
    int m_threshold{0};

  public:
  // weight passed need i0 placement for bias input weights
    Perceptron(Matrix<double, 4, 2> inputs, Matrix<double, 4, 1> targets);

    void pcntrain(Matrix<double, 4, 2> inputs, Matrix<double, 4, 1> targets, double eta, int nIterations);

    Matrix<double, 4, 1> pcnfwd(Matrix<double, 4, 3> inputs);
  };

}
#endif // ML_PERCEPTRON_H
