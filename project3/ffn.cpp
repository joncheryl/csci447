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
    nNodes << 2 << 6 << 6 << 1 << endr ;
    
    // weight matrices
    field<mat> W(nLayers);

    int i;
    for(i=0; i<nLayers; i++){
	W(i) = randn(nNodes(i+1), nNodes(i));
    }
    
/*    mat D;

    D.load("data.csv", csv_ascii);
*/	
    cout << W << endl;

    nNodes.transform( [](double val) { return actFun(val); } );
    
    cout << nNodes << endl;
    
    return 0;
}
