#include "TrainingData.h"
#include "io_utility/io_utility.h"

using namespace io_utility;

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