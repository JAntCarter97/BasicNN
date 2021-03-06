#include "pch.h"
#include "TrainingData.h"


TrainingData::TrainingData(const string fileName)
{
	m_trainingDataFile.open(fileName.c_str());
}


TrainingData::~TrainingData()
{

}

void TrainingData::getTopology(vector<unsigned> &topology)
{
	string line;
	string label;

	getline(m_trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology:") != 0)
	{
		abort();
	}

	while (!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return;
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
	inputVals.clear();
	string line;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("in:") == 0)
	{
		double oneValue;
		while (ss >> oneValue)
		{
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}


unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
	targetOutputVals.clear();

	string line;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("out:") == 0)
	{
		double oneValue;
		while (ss >> oneValue)
		{
			targetOutputVals.push_back(oneValue);
		}
	}
	return targetOutputVals.size();
}