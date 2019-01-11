// BasicNN.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "Net.h"
#include <iostream>
#include <vector>

using std::vector;
using std::cout;
using std::endl;

int main()
{
	vector<unsigned> topology;
	//Creates a 3 -> 2 -> 1 Net with 3 being the inputs, 
	//2 being the hidden layer, and 1 the output
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);
	Net myNet(topology);

	vector<double> inputVals;
	myNet.feedForward(inputVals);

	vector<double> targetVals;
	myNet.backProp(targetVals);

	vector<double> resultVals;
	myNet.getResults(resultVals);
}

