// OptimizationTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <dlib\optimization.h>

typedef dlib::matrix<double, 0, 1> column_vector;

double rosen(const column_vector &m)
{
	const double x = m(0);
	const double y = m(1);
	return 100.0*std::pow(y - x*x, 2) + std::pow(1 - x, 2);
}

int main()
{
	// starting point 
	column_vector x0(2);
	x0 = 29, 199;

	// optimize
	dlib::find_min_using_approximate_derivatives(
		dlib::bfgs_search_strategy(),
		dlib::objective_delta_stop_strategy(1e-9),
		rosen, x0, -1);
	std::cout << "rosen solution: " << x0 << "\n";

	std::system("pause");
    return 0;
}

