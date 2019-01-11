#pragma once

#include "Neuron.h"
#include <vector>
using std::vector;

typedef vector<Neuron> Layer;

class Net
{
public:
	Net(const vector<unsigned> &topology);
	~Net();

	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;

private:
	vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
};

