#include "TrainingData.h"
#include "io_utility/io_utility.h"

using namespace io_utility;

namespace ML::DataSet
{

  void trainPima()
  {
    // Todo read the pima datafile into the data for pcn
    // trainset, targetset,
    MatrixXd dataSet = readDataFile("../dataset/pima-indians-diabetes.data");
    std::cout << dataSet.innerSize() << " " << dataSet.outerSize() << std::endl;

    MatrixXd trainTargets = dataSet(seqN(1, dataSet.innerSize() / 2, 2), last);
    MatrixXd testTargets = dataSet(seqN(0, dataSet.innerSize() / 2, 2), last);
    std::cout << trainTargets.innerSize() << " " << trainTargets.outerSize() << std::endl;
    std::cout << testTargets.innerSize() << " " << testTargets.outerSize() << std::endl;

    MatrixXd trainInputs = dataSet(seqN(1, dataSet.innerSize() / 2, 2), seqN(0, dataSet.outerSize() - 1));
    MatrixXd testInputs = dataSet(seqN(0, dataSet.innerSize() / 2, 2), seqN(0, dataSet.outerSize() - 1));
    std::cout << trainInputs.innerSize() << " " << trainInputs.outerSize() << std::endl;
    std::cout << testInputs.innerSize() << " " << testInputs.outerSize() << std::endl;

    // //  # Perceptron training on the original pima dataset
    double learningRateETA = 0.25;
    Perceptron pcn(trainInputs, trainTargets);
    pcn.pcntrain(trainInputs, trainTargets, learningRateETA, 100);

    pcn.confmat(testInputs, testTargets);
  }

  void testTrainNClasses()
  {
    double learningRateETA = 0.25;

    MatrixXd trainInputs(20, 10);
    trainInputs.fill(0.0);
    MatrixXd trainTargets(20, 10);
    trainTargets.fill(0.0);

    for (int i = 0; i < 10; i++) {
      trainInputs(i, i) = 1.0;
    }
    for (int i = 0; i < 10; i++) {
      trainInputs(i+10, i) = 1.0;
    }
    for (int i = 0; i < 10; i++) {
      trainTargets(i, i) = 1.0;
    }
    for (int i = 0; i < 10; i++) {
      trainTargets(i+10, i) = 1.0;
    }

    std::cout << "Train inputs" << std::endl;
    std::cout << trainInputs << std::endl;
    std::cout << "Train targets" << std::endl;
    std::cout << trainTargets << std::endl;

    Perceptron pcn(trainInputs, trainTargets);
    pcn.pcntrain(trainInputs, trainTargets, learningRateETA, 6);

    pcn.confmat(trainInputs, trainTargets);
  }

  void trainOr()
  {
    double learningRateETA = 0.25;

    MatrixXd trainInputs(4, 2);
    trainInputs << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0;

    MatrixXd trainTargets(4, 1);
    trainTargets << 0.0, 1.0, 1.0, 1.0;

    Perceptron pcn(trainInputs, trainTargets);
    pcn.pcntrain(trainInputs, trainTargets, learningRateETA, 6);

    pcn.confmat(trainInputs, trainTargets);
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

    pcn.confmat(trainInputs, trainTargets);
  }

  MatrixXd readDataFile(std::string fileName)
  {
    MatrixXd dataSet(100, 10);
    dataSet.fill(1.1);
    int countSet = 0;
    int inputCount = 0;
    char ch = ',';

    std::vector<std::vector<double>> dataList;

    // pima = np.loadtxt('pima-indians-diabetes.data',delimiter=',')
    // np.shape(pima)
    // int dataSize;
    // int recordLength;

    auto [dataSize, recordLength] = readDataShapeFromFile(fileName);
    std::cout << "Data file size: " << dataSize << " " << recordLength << std::endl;

    dataSet = MatrixXd(dataSize, recordLength);

    auto fileLines = read_file(fileName, true);
    for (std::string fileLine : fileLines)
    {
      auto stripLine = stripFileLine(fileLine);
      if (isCommentLine(stripLine))
      {
        std::cout << stripLine << std::endl;
        continue;
      }
      auto dataLineArray = splitStringToDouble(stripLine, ch);
      inputCount = dataLineArray.size();
      if (inputCount > 0)
      {
        for (int i = 0; i < inputCount; i++)
        {
          dataSet(countSet, i) = dataLineArray[i];
        }
      }
      countSet++;
    }

    return dataSet;
  }

  std::string stripFileLine(std::string line)
  {
    char ch = '\r';
    std::string stripLine("");
    auto it = std::find(line.begin(), line.end(), ch);
    stripLine.resize(std::distance(line.begin(), it));
    std::copy(line.begin(), it, stripLine.begin());
    return stripLine;
  }

  bool isCommentLine(std::string line)
  {
    char ch = '#';
    bool isCommentLine = false;
    auto it = std::find(line.begin(), line.end(), ch);
    if (it != line.end())
    {
      isCommentLine = true;
    }
    return isCommentLine;
  }

  std::vector<double> splitStringToDouble(const std::string &str, char delimiter)
  {
    std::vector<double> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter))
    {
      tokens.push_back(std::stod(token));
    }

    return tokens;
  }

  std::tuple<int, int> readDataShapeFromFile(std::string fileName)
  {
    int countSet = 0;
    int inputCount = 0;
    char ch = ',';
    auto fileLines = read_file(fileName, true);

    for (std::string fileLine : fileLines)
    {
      auto stripLine = stripFileLine(fileLine);
      if (isCommentLine(stripLine))
      {
        std::cout << stripLine << std::endl;
        continue;
      }
      auto dataLineArray = splitStringToDouble(stripLine, ch);
      if (inputCount > 0)
      {
        if (dataLineArray.size() != inputCount)
        {
          std::cout << "Found discripancy in data input line sizes!" << std::endl;
        }
      }
      inputCount = dataLineArray.size();
      countSet++;
    }

    return {countSet, inputCount};
  }


}