/*
 *
 * Compile with:
 * g++ -std=c++11 ffn.cpp -o ffn -O2 -larmadillo
 *
 */

#include <iostream>
#include <armadillo>
#include <cmath>

#include <iterator>
#include <vector>
#include <algorithm>

using namespace std;
using namespace arma;

/*
 *
 * Activation Functions
 *
 * Should these take and return doubles or should they take and return vectors?
 *
 */
double actFun(double x, double act = 0)
{
    if(act == 0)  // Use sigmoid activation function
    {
        if(x > 0) return (1 / (1 + exp(-x)));
	else return (exp(x) / (1 + exp(x)));
    }
    else  // Use linear activation function
    {
	return act * x;
    }
}

double dActFun(double x, double act = 0)
{
    if(act == 0)  // Use sigmoid activation function
    {
        if(x > 0) return ( exp(-x) / ((1 + exp(-x))*(1 + exp(-x))) );
        else return ( exp(x) / ((1 + exp(x))*(1 + exp(x))) );
    }
    else  // Use linear activation function
    {
        return act;
    }
}


/*
 *
 * Program
 *
 */
int main(int argc, char** argv)
{
    // number of hidden layers + 1
    int nLayers = 3;

    // learning rate
    double lRate = .05;
    
    // array containing number of nodes in each layer (first and last element
    // are the number of inputs and outputs respectively.
    vec nNodes;
    nNodes << 3 << 6 << 6 << 2 << endr ;
    
    // weight and bias matrices
    field<mat> W(nLayers);
    field<vec> B(nLayers);
    int i;
    for(i=0; i<nLayers; i++){
	W(i) = randn(nNodes(i+1), nNodes(i));
	B(i) = randn(nNodes(i+1));
    }

    // net input and activation vectors
    field<vec> netInputs(nLayers);
    field<vec> activations(nLayers+1);

    // input training data
    mat D;
    D.load("data.csv", csv_ascii);

    mat targets;
    targets.load("data.csv", csv_ascii);
    targets = targets.cols(0, 1);
    
    // feedforward
    vec activation = D.row(0).t();
    activations(0) = activation;
    vec netInput;

    // possible to remove some of these intermediate containers?
    // i.e. do i need activation below
    for(i=0; i<nLayers; i++)
    {
        netInput = W(i) * activation + B(i);
        netInputs(i) = netInput;
	activation = netInput.transform([](double val) { return actFun(val); });
	activations(i+1) = activation;
    }

    // backprop it
    field<mat> deltaW(nLayers);
    field<mat> deltaB(nLayers);

    // get target value
    vec y = targets.row(0).t();

    // initialize backprop process
    deltaB(nLayers - 1) = (activations(nLayers) - y) % netInputs(nLayers-1).transform([](double val) { return dActFun(val); });
    deltaW(nLayers - 1) = deltaB(nLayers - 1) * activations(nLayers - 1).t();

    // throw back errors
    for(i = nLayers - 2; i >= 0; i--)
    {
	deltaB(i) = W(i + 1).t() * deltaB(i + 1) % netInputs(i).transform([](double val) { return dActFun(val); });
	deltaW(i) = deltaB(i) * activations(i).t();
    }
    
    // adjust with learning rate
    for(i = 0; i < nLayers; i++)
    {
	deltaW(i).transform([=](double val) { return lRate * val; });
	deltaB(i).transform([=](double val) { return lRate * val; });
    }
    
    cout << deltaW <<endl;
    
    return 0;
}
