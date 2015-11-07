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
 */
double actFun(double x)
{
    double act = 0;
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

    // array containing number of nodes in each layer (first and last element
    // are the number of inputs and outputs respectively.
    vec nNodes;
    nNodes << 3 << 6 << 6 << 1 << endr ;
    
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

    // feedforward
    vec activation = D.row(0).t();
    activations(0) = activation;
    vec netInput;

    // possible to remove some of these intermediate containers?
    // i.e. do i need activation below
    for(i=0; i<nLayers; i++){
        netInput = W(i) * activation + B(i);
        netInputs(i) = netInput;
	activation = netInput.transform([](double val) { return actFun(val); });
	activations(i+1) = activation;
    }

    // backprop it
    
    cout << activations <<endl;
    
    return 0;
}
