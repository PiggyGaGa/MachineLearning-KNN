#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cmath>

using std::cout; using std::cin; using std::cerr; using std::endl;
using std::string; using std::ifstream; using std::istringstream;
using std::vector; using std::ios;

namespace machinelearning
{
	class KNN
	{
	private:
		cv::Mat trainData;
		cv::Mat trainLabel;
		cv::Mat nearestNeighborsLabel;
		cv::Mat nearestDistance;
		template <typename Type> Type StringToNum(const string& str);  //from string type to numerical type;
		float EuclidenDistance(cv::Mat x, cv::Mat y);
		int IfSmallerThanExist(float distance, vector<float> kDistance);
		cv::Mat FindReaultFromNearestNeighbor(cv::Mat &nearestNeighbors, int k);

	public:
		KNN();
		int LoadTrainDataSet(string fileName, cv::Mat &trainDataSet, cv::Mat &trainLabelSet); //load train samples from csv file
		int LoadTestDataSet(string fileName, cv::Mat &testDataSet);
		int Train(cv::Mat &trainData, cv::Mat &trainLabel);
		int FindNearestNeighbor(int k, cv::Mat &InputSample, cv::Mat &result, string distanceType);
		cv::Mat GetNearestNeighbors();
		cv::Mat GetNearestDistance();
		~KNN();
	};

}

namespace machinelearning
{
	KNN::KNN()
	{

	}
	int KNN::LoadTrainDataSet(string fileName, cv::Mat &trainDataSet, cv::Mat &trainLabelSet)
	{
		class LabelInfo
		{
		public:
			struct stringInt
			{
				int intValue;
				string stringValue;
			};
		    int StringIfInVector(string a, vector<stringInt> b)
			{
				for (int i = 0; i < b.size(); i++)
				{
					if (a == (b.begin() + i)->stringValue)
					{
						return (b.begin() + i)->intValue;
					}
				}
				return -1;
			}
		};
			
		vector<LabelInfo::stringInt> labelInfo;
		ifstream read;
		read.open(fileName, ios::in);
		if (!read)
		{
			cerr << "The DataSet file does not exist" << endl;
			getchar();
			exit(0);
		}
		string line;
		while (getline(read, line))
		{
			vector<float> oneLineNum;  //storage one line number;
			vector<string> oneLineString;
			istringstream lineString(line);  // one line String stream;
			char dim = ',';
			string singleStringNum;
			while (getline(lineString, singleStringNum, dim))
			{
				oneLineString.push_back(singleStringNum);
				float tmpNum = StringToNum<float>(singleStringNum);
				oneLineNum.push_back(tmpNum);
			}
			cv::Mat oneLineNumMat(1, int(oneLineString.size() - 1), CV_32FC1);
			vector<float>::iterator iter1 = oneLineNum.begin(), iter2 = oneLineNum.end();
			int i = 0;
			while (iter1 < iter2 - 1)
			{
				oneLineNumMat.at<float>(0, i) = *iter1;
				iter1++;
				i++;
			}
			trainDataSet.push_back(oneLineNumMat);
			LabelInfo fun;
			int index = fun.StringIfInVector(*(oneLineString.end() - 1), labelInfo);
			if (index >= 0)
			{
				trainLabelSet.push_back((labelInfo.begin() + index)->intValue);
			}
			else
			{
				LabelInfo::stringInt tmp;
				tmp.intValue = int(labelInfo.size());
				tmp.stringValue = *(oneLineString.end() - 1);
				labelInfo.push_back(tmp);
				trainLabelSet.push_back(tmp.intValue);
			}
		}
		return 0;
	}

	int KNN::LoadTestDataSet(string fileName, cv::Mat &testDataSet)
	{
		ifstream read;
		read.open(fileName);
		if (!read)
		{
			cerr << "Can not open test data file" << endl;
			system("pause");
			exit(0);
		}
		string line;
		int length = 0;
		getline(read, line);
		istringstream lineStringStream(line);
		string oneString;
		char dim = ',';
		vector<string> stringVec;
		cv::Mat lineFloat;
		while (getline(lineStringStream, oneString, dim))
		{
			stringVec.push_back(oneString);
			lineFloat.push_back(StringToNum<float>(oneString));
		}
		length = int(stringVec.size());
		testDataSet.push_back(lineFloat.t());
		while (getline(read, line))
		{
			istringstream lineStringStream(line);
			string oneString;
			char dim = ',';
			vector<string> stringVec;
			cv::Mat lineFloat;
			while (getline(lineStringStream, oneString, dim))
			{
				stringVec.push_back(oneString);
				lineFloat.push_back(StringToNum<float>(oneString));
			}
			if (stringVec.size() == length)
			{
				testDataSet.push_back(lineFloat.t());
			}
		}
		return 0;
	}
	int KNN::Train(cv::Mat &trainDataSet, cv::Mat &trainLabelSet)
	{
		trainLabelSet.convertTo(trainLabelSet, CV_32SC1);
		this->trainData = trainDataSet;
		this->trainLabel = trainLabelSet;
		return 1;
	}

	int KNN::FindNearestNeighbor(int k, cv::Mat &InputSample, cv::Mat &result, string distanceType)
	{
		cv::Mat_<int> re(InputSample.rows, 1);   // storage answer;
		cv::Mat_<float> nearestDistance(k * InputSample.rows , 1);  //NearestNeighbors is a k*Sample_dimension, NuearesDistance
		cv::Mat_<int> nearestLabel(k * InputSample.rows, 1);
		if (InputSample.cols != this->trainData.cols)
		{
			cerr << "The dimesion of InputData is not match with trainData" << endl;
			system("pause");
			exit(0);
		}
		if (this->trainData.rows != this->trainLabel.rows)
		{
			cerr << "The dimension of train  data is not match with train label" << endl;
			system("pause");
			exit(0);
		}
		for (int sampleIndex = 0; sampleIndex < InputSample.rows; sampleIndex++)
		{  // for each sample find k nearest neighnbors;
			vector<float> kDistance; // k nearest's distance
			for (int i = 0; i < k; i++)
			{
				float distance = EuclidenDistance(this->trainData.row(i), InputSample.row(sampleIndex));
				kDistance.push_back(distance);
				nearestLabel.at<int>(sampleIndex * k + i, 0) = this->trainLabel.at<int>(i, 0);
			}
			for (int i = k; i < this->trainData.rows; i++)
			{
				float distance = EuclidenDistance(this->trainData.row(i), InputSample.row(sampleIndex));
				int maxIndex = IfSmallerThanExist(distance, kDistance);
				if (maxIndex >=0)
				{
					*(kDistance.begin() + maxIndex) = distance;
					nearestLabel.at<int>(sampleIndex * k + maxIndex, 0) = this->trainLabel.at<int>(i, 0);
				}
			}
			for (int i = 0; i < k; i++)
			{
				nearestDistance.at<float>(sampleIndex * k + i, 0) = kDistance[i];
			}
		}
		re = FindReaultFromNearestNeighbor(nearestLabel, k);
		result = re;
		return 0;
	}

	KNN::~KNN()
	{
		this->nearestDistance.release();
		this->nearestNeighborsLabel.release();
		this->trainData.release();
		this->trainLabel.release();
	}

	float KNN::EuclidenDistance(cv::Mat x, cv::Mat y)
	{
		if (x.rows > 1 || y.rows > 1)
		{
			cerr << "Dimension of input data is more than 1" << endl;
			system("pause");
			exit(0);
		}
		if (x.cols != y.cols)
		{
			cerr << "the dimension of x is not match with y" << endl;
			system("pause");
			exit(0);
		}
		float sum = 0.0;
		for (int i = 0; i < x.cols; i++)
		{
			sum = sum + pow((x.at<float>(0, i) - y.at<float>(0, i)), 2);
		}
		return sqrt(sum);
	}
	int KNN::IfSmallerThanExist(float distance, vector<float> kDistance)
	{//if value is smaller than the value in kDistance, if true return the max value's  index in kDistance
		int flag = 0, maxIndex;
		float maxValue = 0;
		for (int i = 0; i < kDistance.size(); i++)
		{
			if (distance <= *(kDistance.begin() + i))
			{
				flag++;
			}
			if (*(kDistance.begin() + i) >= maxValue)
			{
				maxIndex = i;
			}
		}
		if (flag == kDistance.size())
		{
			return maxIndex;
		}
		return -1;
	}
	cv::Mat KNN::GetNearestNeighbors()
	{
		return this->nearestNeighborsLabel;
	}
	cv::Mat KNN::GetNearestDistance()
	{
		return this->nearestDistance;
	}
	cv::Mat KNN::FindReaultFromNearestNeighbor(cv::Mat &nearestLabel, int k)
	{
		class TM
		{
		public:
			struct tmpstruct
			{
				int value;
				int num;
			};
			static int IntIfInVector(int a, vector<tmpstruct> b)
			{
				for (int i = 0; i < b.size(); i++)
				{
					if (a == b[i].value)
					{
						return i;
					}
				}
				return -1;
			}
		};
		cv::Mat_<int> result;
		int sampleNum = nearestLabel.rows / k;
		for (int i = 0; i < sampleNum; i++)
		{
			vector<TM::tmpstruct> frequencyInfomation;
			for (int j = 0; j < k; j++)
			{
				int index = TM::IntIfInVector(int(nearestLabel.at<int>(i * k + j, 0)), frequencyInfomation);
				if (index >= 0)
				{
					frequencyInfomation[index].num++;
				}
				else
				{
					TM::tmpstruct c;
					c.num = 1;
					c.value = int(nearestLabel.at<int>(i * k + j, 0));
					frequencyInfomation.push_back(c);
				}
			}
			int max = 0, maxNum = 0;
			for (int j = 0; j < frequencyInfomation.size(); j++)
			{
				if (frequencyInfomation[j].num > maxNum)
				{
					max = frequencyInfomation[j].value;
					maxNum = frequencyInfomation[j].num;
				}
			}			
			result.push_back(max);
			//cout << "type()" << endl;
			//cout << result.type();
		}
		return cv::Mat(result);
	}
	//模板函数：将string类型变量转换为常用的数值类型（此方法具有普遍适用性）  
	template <typename Type>
	Type KNN::StringToNum(const string& str)
	{
		istringstream iss(str);
		Type num;
		iss >> num;
		return num;
	}
}