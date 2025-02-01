#include "Perceptron.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <cmath>

#ifdef __STDC_IEC_559__
constexpr double HasStd = 1.000023;
#else
constexpr double HasStd = 1.0;
#endif

namespace ML
{
  Perceptron::Perceptron(Matrix<double, 4, 2> inputs, Matrix<double, 4, 1> targets)
      : m_nIn{1}, m_nOut{1}, m_nData{0}
  {

    // Set up network size
    int nIn = 1;
    int nOut = 1;
    if (inputs.NumDimensions > 1)
    {
      m_nIn = inputs.outerSize();
    }

    if (targets.NumDimensions > 1)
    {
      m_nOut = inputs.outerSize();
    }

    m_nData = inputs.innerSize();

    std::cout << "network size " << std::endl
              << " nIn: " << m_nIn << ", nOut:" << m_nOut << ", nData: " << m_nData << std::endl;

    // # Initialise network
    ArrayXd a2 = ArrayXd::Random((m_nIn + 1) * m_nOut);
    m_weights.resize(m_nIn + 1, m_nOut);
    m_weights <<  a2 * (0.1 - 0.05);

    std::cout << "random weights in network initialized: " << m_weights << std::endl;
  }

  void Perceptron::pcntrain(Matrix<double, 4, 2> inputs, Matrix<double, 4, 1> targets, double eta, int nIterations)
  {
    Matrix<double, 4, 1> biasInput;
    biasInput << -1.0, -1.0, -1.0, -1.0;
    Matrix<double, 4, 3> inputsWithBiasEntry;
    inputsWithBiasEntry.block(0, 0, 4, 2) << inputs;
    inputsWithBiasEntry.col(2).tail(4) << biasInput; //-1, -1, -1, -1;

    std::cout << "train inputs: " << std::endl
              << inputs << std::endl;
    std::cout << "train inputs with bias: " << std::endl
              << inputsWithBiasEntry << std::endl;
    std::cout << "train targets " << std::endl
              << targets << std::endl;

    for (int i = 0; i < nIterations; i++)
    {
      Matrix<double, 4, 1> thresholdactivations = pcnfwd(inputsWithBiasEntry); 
      std::cout << "Threshold activations at iter:" << i << " " << "\n"
                << thresholdactivations << std::endl;
      auto transposeInputs = inputsWithBiasEntry.transpose();
      std::cout << "transpose inputs: " << i << " " << std::endl
                << transposeInputs << std::endl;
      m_weights -= eta * (transposeInputs * (thresholdactivations - targets));
      std::cout << "train weights at iter: " << i << " " << std::endl
                << m_weights << std::endl;
    }
  }
  // This function does the recall computation
  // Eigen::ArrayXXd result = (activations.array() >= 26).select(a, b);
  // Matrix<double, 2, 2> result2 = (Nresults.array() >= 26).select(a, b);
  /**
   * the folloing is using eigen matrix dot function, which can only work with vectors
   *
  for (int i = 0; i < trainInputs.innerSize(); i++)
  {
    std::cout << "[" << trainInputs.row(i) << "] " << std::endl;
    for (int j = 0; j < trainWeights.outerSize(); j++)
    {
      std::cout << "[" << trainWeights.col(j) << "] - ";
      std::cout << std::setw(5) << trainInputs.row(i).dot(trainWeights.col(j)) << " " << std::endl;
      activations(i, j) = trainInputs.row(i).dot(trainWeights.col(j));
    }
    std::cout << std::endl;
  }
  std::cout << "Complete activation: " << activations << std::endl;

  */
  Matrix<double, 4, 1> Perceptron::pcnfwd(Matrix<double, 4, 3> inputs)
  {
    Eigen::ArrayXXd a(4, 1);
    a.fill(1.0);
    Eigen::ArrayXXd b(4, 1);
    b.fill(0);
    // We use matrix multiplication instead of seperating into vectors and using dot product to build up the mtrix for
    // activations for neurons. It is what is happening with that dot method to get sum product of two vectors.
    // We can use matrix mult, only when inner sizes are the same
    Matrix<double, 4, 1> Nresults{inputs * m_weights};
    // auto Nresults{inputs * weights};
    std::cout << "inputs * weights: " << std::endl
              << Nresults << std::endl;

/////////
    // Matrix<double, 4, 1> activations;
    // for (int i = 0; i < inputs.innerSize(); i++)
    // {
    //   std::cout << "[" << inputs.row(i) << "] " << std::endl;
    //   for (int j = 0; j < m_weights.outerSize(); j++)
    //   {
    //     std::cout << "[" << m_weights.col(j) << "] - ";
    //     std::cout << std::setw(5) << inputs.row(i).dot(m_weights.col(j)) << " " << std::endl;
    //     activations(i, j) = inputs.row(i).dot(m_weights.col(j));
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "Complete activation: " << activations << std::endl;
    // std::cout << "Complete activation: " << (activations.array() > m_threshold).select(a,b) << std::endl;
    ////////

    // Select the threshold
    return (Nresults.array() > m_threshold).select(a, b);
  }
}