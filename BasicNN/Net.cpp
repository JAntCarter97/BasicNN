#include "pch.h"
#include "Net.h"

#include <iostream>
using std::cout;
using std::endl;
using std::vector;


Net::Net(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();

	//Outer Loop creates a new layer
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++)
	{
		m_layers.push_back(Layer());

		//Inner loop goes through each layer and adds neurons to the layer
		//Includes bias
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
		{
			m_layers.back().push_back(Neuron());
			cout << "Made a Neuron" << endl;
		}
	}
}


Net::~Net()
{
	//Empty
}



void Net::feedForward(const vector<double> &inputVals)
{
	
}

void Net::backProp(const vector<double> &targetVals)
{

}

void Net::getResults(vector<double> &resultVals) const 
{

}