#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
using std::vector;
using std::string;

class TrainingData
{
public:
	TrainingData(const string fileName);
	~TrainingData();

	bool isEof(void) { return m_trainingDataFile.eof(); }
	void getTopology(vector<unsigned> &topology);

	//Returns the number of input values read from the file
	unsigned getNextInputs(vector<double> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
	std::ifstream m_trainingDataFile;
};

