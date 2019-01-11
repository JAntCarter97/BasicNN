#include "pch.h"
#include "Neuron.h"

#include <cmath>

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; c++)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}


Neuron::~Neuron()
{
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
	// Include the bias Node from the previous layer

	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_ouputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

double Neuron::transferFunction(double x)
{
	// tanh - output range [-1.0, 1.0]
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	// tanh derivative
	return 1.0 - x * x;
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer)
{
	double sum = 0.0;
	
	//Sum our contributions of the errors at the nodes we feed
	for (unsigned n = 0; n < nextLayer.size() - 1; n++)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

double Neuron::eta = 0.15; //Overall net learning rate
double Neuron::alpha = 0.5; // Momentum, multiplier of last deltaWeight

void Neuron::updateInputWeights(Layer &prevLayer)
{
	// THe weights to the updated are in the connection container 
	// in the neurons in the preceding layer
	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}