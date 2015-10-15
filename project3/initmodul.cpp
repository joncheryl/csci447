#include "CPPModulWrapper.hpp"

static PyMethodDef methodtable[] = {
    {"fibun", wrap_fibun, METH_VARARGS, "int fibun(int)" },
    {NULL, NULL}
};

// unix/linux:
extern "C" {

    void initCPPModul() {
	PyObject* m = Py_InitModule4("CPPModul",
				     methodtable,
				     "fibun",
				     0,
				     PYTHON_API_VERSION);
    }
}
