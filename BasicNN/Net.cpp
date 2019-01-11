#include "pch.h"
#include "Net.h"

#include <iostream>
#include <cassert>
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

		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum - 1];

		//Inner loop goes through each layer and adds neurons to the layer
		//Includes bias
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
		{
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
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
	assert(inputVals.size() == m_layers[0].size() - 1);

	//Connect input values to the input neurons
	for (unsigned i = 0; i < inputVals.size(); i++)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	//Forward Prop
	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++)
	{
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++)
		{
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const vector<double> &targetVals)
{
	//Calc overall net error
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; n++)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
	}

	m_error /= outputLayer.size() - 1; // Get average error squared
	m_error = sqrt(m_error); // RMS

	//Implement a recent average measurement.
	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);

	//Calc output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; n++)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	//Calc gradients on hidden layers
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); n++)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	//For all layers from outputs to first hidden layer
	// update connection weights.
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; n++)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::getResults(vector<double> &resultVals) const 
{
	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size(); n++)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}