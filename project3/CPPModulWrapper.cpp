#include "CPPModulWrapper.hpp"
#include "CPPModul.hpp"

PyObject* wrap_fibun(PyObject*, PyObject* args)
{
    int input = 0;
    if(!PyArg_ParseTuple(args, "i", &input))
	return NULL;

    return Py_BuildValue("i", fibun(input));
}
