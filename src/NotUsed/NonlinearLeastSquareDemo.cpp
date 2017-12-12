// NonlinearLeastSquareDemo.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <dlib\optimization.h>
#include <iostream>
#include <vector>

typedef dlib::matrix<double, 2, 1> input_vector;
typedef dlib::matrix<double, 3, 1> parameter_vector;

double model(const input_vector &input,
	const parameter_vector &params)
{
	const double p0 = params(0);
	const double p1 = params(1);
	const double p2 = params(2);

	const double i0 = input(0);
	const double i1 = input(1);

	const double temp = p0*i0 + p1*i1 + p2;

	return temp*temp;
}

double residual(const std::pair<input_vector, double> &data,
	const parameter_vector &params)
{
	
	return model(data.first, params) - data.second;
}

parameter_vector residual_deriv(const std::pair<input_vector, double> &data,
	const parameter_vector &params)
{
	parameter_vector der;

	const double p0 = params(0);
	const double p1 = params(1);
	const double p2 = params(2);

	const double i0 = data.first(0);
	const double i1 = data.first(1);

	const double temp = p0*i0 + p1*i1 + p2;

	der(0) = i0 * 2 * temp;
	der(1) = i1 * 2 * temp;
	der(2) = 2 * temp;

	return der;
}


int main()
{
	// random define the parameters
	parameter_vector params = 10 * dlib::randm(3, 1);
	std::cout << "params: " << dlib::trans(params) << std::endl;

	// generate data
	std::vector<std::pair<input_vector, double>> data_samples;
	input_vector input;
	for (int i=0; i<1000;++i)
	{
		input = 10 * dlib::randm(2, 1);
		const double output = model(input, params);
		data_samples.push_back(std::make_pair(input, output));
	}

	// check deriv
	std::cout << "derivative error: " << 
		dlib::length(residual_deriv(data_samples[0], params) -
		dlib::derivative(residual)(data_samples[0], params)) << std::endl;

	// solve
	parameter_vector x;
	x = 1;

	std::cout << "Use Levenberg-Marquardt" << std::endl;
	// Use the Levenberg-Marquardt method to determine the parameters which
	// minimize the sum of all squared residuals.
	dlib::solve_least_squares_lm(
		dlib::objective_delta_stop_strategy(1e-7).be_verbose(),
		residual,
		residual_deriv,
		data_samples,
		x);

	// Now x contains the solution.  If everything worked it will be equal to params.
	std::cout << "inferred parameters: " << trans(x) << std::endl;
	std::cout << "solution error:      " << length(x - params) << std::endl;
	std::cout << std::endl;

	std::system("pause");
    return 0;
}

