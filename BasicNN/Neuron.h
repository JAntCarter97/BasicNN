#pragma once

#include <vector>
#include <cstdlib>

using std::vector;

typedef vector<Neuron> Layer;

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

	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	
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
	double sumDOW(const Layer &nextLayer);
};

