#include "CPPModul.hpp"

int fibun(int n)
{
    if(n > 2)
	return fibun(n - 1) + fibun(n - 2);
    return 1;
}
