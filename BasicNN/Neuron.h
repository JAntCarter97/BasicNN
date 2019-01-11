#pragma once

#include <vector>
#include <cstdlib>

using std::vector;

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	~Neuron();

	void feedForward(const vector<Neuron> &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const vector<Neuron> &nextLayer);
	void updateInputWeights(vector<Neuron> &prevLayer);
	
	//Getters Setters
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }

private:
	double m_outputVal;
	unsigned m_myIndex;
	double m_gradient;
	static double eta;
	static double alpha;
	vector<Connection> m_outputWeights;
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double sumDOW(const vector<Neuron> &nextLayer);
};

