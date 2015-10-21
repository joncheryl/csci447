#include "CPPModulWrapper.hpp"
#include "CPPModul.hpp"
#include <stdlib.h>
#include <string>
#include <sstream>

PyObject* wrap_fibun(PyObject*, PyObject* args)
{
    int numLines;       /* how many lines we passed for parsing */
    char * line;        /* pointer to the line as a string */
    char * token;       /* token parsed by strtok */

    PyObject * listObj; /* the list of strings */
    PyObject * strObj;  /* one string in the list */

    // the O! parses for a Python object (listObj) checked to be of type PyList_Type
    if (! PyArg_ParseTuple( args, "O!", &PyList_Type, &listObj))
	return NULL;

    // get the number of lines passed to us
    numLines = PyList_Size(listObj);

    double stuffy[numLines];

    // should raise an error here.
    if (numLines < 0)   return NULL; // Not a list
    
    // iterate over items of the list, grabbing doubles, and parsing for numbers
    int i;
    for (i=0; i<numLines; i++){

        // grab the string object from the next element of the list
	strObj = PyList_GetItem(listObj, i); // Can't fail

        // make it a string
	stuffy[i] = PyFloat_AsDouble( strObj );
    }

    double summation = 0;

    for (i=0; i < numLines; i++) summation += stuffy[i];
    
    return Py_BuildValue("d", summation);
}
