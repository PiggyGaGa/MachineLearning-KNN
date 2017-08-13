# include "KNNClass.h"

using namespace std;

int main()
{
	machinelearning::KNN myKNN;
	cv::Mat trainDataSet, trainLabelSet;
	cv::Mat InputSample;
	cv::Mat result;
	myKNN.LoadTrainDataSet("Iris.csv", trainDataSet, trainLabelSet);
	myKNN.Train(trainDataSet, trainLabelSet);
	myKNN.LoadTestDataSet("IrisTest.csv", InputSample);
//	cout << InputSample << endl;
//	cout << trainDataSet << endl;
//	cout << trainLabelSet << endl;
	myKNN.FindNearestNeighbor(3, InputSample, result, "Eu");
	cout << result << endl;
	system("pause");

}