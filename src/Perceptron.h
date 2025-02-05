#ifndef ML_PERCEPTRON_H
#define ML_PERCEPTRON_H

#include <Eigen/Dense>

using namespace Eigen;

#ifdef NDEBUG
#define D(x)
#else
#define D(x) x
#endif

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
    Perceptron(MatrixXd inputs, MatrixXd targets);

    void pcntrain(MatrixXd inputs, MatrixXd targets, double eta, int nIterations);

    MatrixXd pcnfwd(MatrixXd inputs);

    void confmat(MatrixXd inputs, MatrixXd targets);
  };

}
#endif // ML_PERCEPTRON_H
