// BasicNN.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "Net.h"
#include "TrainingData.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <fstream>
#include <sstream>

using std::vector;
using std::cout;
using std::endl;
using std::string;

//Function declarations
void showVectorVals(string label, vector<double> &v);

int main()
{
	TrainingData trainData("trainingData.txt");

	vector<unsigned> topology;
	trainData.getTopology(topology);
	Net myNet(topology);

	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;

	while (!trainData.isEof())
	{
		trainingPass++;
		cout << endl << "Pass " << trainingPass;

		//Get new input data and feed it forward.
		if (trainData.getNextInputs(inputVals) != topology[0])
		{
			break;
		}
		showVectorVals(": Inputs:", inputVals);

		myNet.feedForward(inputVals);

		myNet.getResults(resultVals);
		showVectorVals("Outputs: ", resultVals);

		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets: ", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		//Report how well the training is working
		cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
	}

	cout << endl << "Done" << endl;
}

void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); i++)
	{
		cout << v[i] << " ";
	}

	cout << endl;
}

