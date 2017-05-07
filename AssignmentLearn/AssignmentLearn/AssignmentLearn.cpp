// AssignmentLearn.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <dlib/svm_threaded.h>

// define types
typedef dlib::matrix<double, 0, 1> col_vec;
// pair def: first is LHS, second is RHS
typedef std::pair<std::vector<col_vec>, std::vector<col_vec>> sample_type;
// associate info between LHS and RHS
typedef std::vector<long> label_type;

// all LHS and RHS are 3-dimensional vector in this example
const unsigned long num_dims = 3;

// pre define
void make_data(std::vector<sample_type> &samples,
	std::vector<label_type> &labels);

// feature extractor
struct feature_extractor
{
	typedef col_vec feature_vector_type;
	typedef col_vec lhs_element;
	typedef col_vec rhs_element;

	unsigned long num_features() const
	{
		return num_dims;
	}

	void get_features(const lhs_element &left,
		const rhs_element &right,
		feature_vector_type &feats) const
	{
		dlib::squared(left - right);
	}
};

// serialization - empty since no state
void serialize(const feature_extractor&, std::ostream&) {}
void deserialize(const feature_extractor&, std::istream&) {}





int main()
{

    return 0;
}

