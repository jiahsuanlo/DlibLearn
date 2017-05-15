// DeepAdvanced.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

// type def block layer
template <
	int N,
	template <typename> class BN,
	int stride,
	typename SUBNET
>
using block = BN<dlib::con<N, 3, 3, stride, stride, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

// define skip/residual layer alias
template<
	template<int, template<typename>class, int, typename> class block,
	int N,
	template<typename> class BN,
	typename SUBNET
>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>; // stride 1

// define residual down-sampling layer
// residual_down creates a network structure like this:
/*
	input from SUBNET
	/     \
	/       \
	block     downsample(using avg_pool)
	\       /
	\     /
	add tensors (using add_prev2 which adds the output of tag2 with avg_pool's output)
	|
	output
*/
template<
	template <int, template<typename>class, int, typename> block,
	int N,
	template<typename> class BN,
	typename SUBNET
>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2,
	dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;


// define 4 different residual block
template <typename SUBNET> using res = relu<residual<block, 8, dlib::bn_con, SUBNET>>;
template <typename SUBNET> using ares = relu<residual<block, 8, dlib::affine, SUBNET>>;
template <typename SUBNET> using res_down = relu<residual_down<block, 8, dlib::bn_con, SUBNET>>;
template <typename SUBNET> using ares_down = relu<residual_down<block, 8, dlib::affine, SUBNET>>;

// define the real network now
const unsigned long number_of_classes = 10;
using net_type = dlib::loss_multiclass_log < dlib::fc < number_of_classes,
	dlib::avg_pool_everything < res < res < res < res_down <
	dlib::repeat < 9, res,
	res_down < res<dlib::input<dlib::matrix<unsigned char>>
	>>>>>>>>>>;

// using parametric relu to design a network
template <tyname SUBNET>
using pres = dlib::prelu < add_prev1<dlib::bn_con<bn::con<8, 3, 3, 1, 1,
	dlib::prelu<dlib::bn_con<dlib::con<8, 3, 3, 1, 1, dlib::tag1<SUBNET>>>>>>>;

// need mnist images to run

int main()
{
	// load data
	std::vector<dlib::matrix<unsigned char>> train_images;
	std::vector<unsigned long> train_labels;
	std::vector<dlib::matrix<unsigned char>> test_images;
	std::vector<unsigned long> test_labels;
	dlib::load_mnist_dataset("../../data/mnistimages", train_images,
		train_labels, test_images, test_labels);

	// set cudnn to use smaller ram required algorithm
	dlib::set_dnn_prefer_smallest_algorithms();



    return 0;
}

