// DeepAdvanced.cpp : Defines the entry point for the console application.
//

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

// type def block layer
// Nx3x3-conv  -> BN -> relu -> Nx3x3-conv -> BN
template <
	int N,
	template <typename> class BN,
	int stride,
	typename SUBNET
>
using block = BN<dlib::con<N, 3, 3, stride, stride, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

// define skip/residual layer alias
/* 
	  subnet input
	   |      |
       \    block
	    \   /
	     add
		  |
		output
*/
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
	template <int, template<typename>class, int, typename> class block,
	int N,
	template<typename> class BN,
	typename SUBNET
>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2,
	dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;


// define 4 different residual layer with and without down-sampling, 
// with and without batch-normalization. Affine version will be used
// when testing the network
template <typename SUBNET> using res = dlib::relu<residual<block, 8, dlib::bn_con, SUBNET>>;
template <typename SUBNET> using ares = dlib::relu<residual<block, 8, dlib::affine, SUBNET>>;
template <typename SUBNET> using res_down = dlib::relu<residual_down<block, 8, dlib::bn_con, SUBNET>>;
template <typename SUBNET> using ares_down = dlib::relu<residual_down<block, 8, dlib::affine, SUBNET>>;

// define the real network now
const unsigned long number_of_classes = 10;
using net_type = dlib::loss_multiclass_log < dlib::fc < number_of_classes,
		dlib::avg_pool_everything < res < res < res < res_down <
			dlib::repeat <9, res,
				res_down < res<dlib::input<dlib::matrix<unsigned char>>
	>>>>>>>>>>;

// using parametric relu to design a network
template <typename SUBNET>
using pres = dlib::prelu < dlib::add_prev1<dlib::bn_con<dlib::con<8, 3, 3, 1, 1,
	dlib::prelu<dlib::bn_con<dlib::con<8, 3, 3, 1, 1, dlib::tag1<SUBNET>  >>>>>>>;

// need mnist images to run

int main()
{
	// load data
	std::vector<dlib::matrix<unsigned char>> train_images;
	std::vector<unsigned long> train_labels;
	std::vector<dlib::matrix<unsigned char>> test_images;
	std::vector<unsigned long> test_labels;
	dlib::load_mnist_dataset("../data/mnistimages", train_images,
		train_labels, test_images, test_labels);

	// set cudnn to use smaller ram required algorithm
	dlib::set_dnn_prefer_smallest_algorithms();

	// create a network
	net_type net;

	// practice: replace relu layers with prelu layers
	using net_type2 = dlib::loss_multiclass_log<dlib::fc<number_of_classes, 
		dlib::avg_pool_everything<
			pres<res<res<res_down<
				dlib::tag4<dlib::repeat<9,pres,
					res_down< 
						res<dlib::input<dlib::matrix<unsigned char>> >>>>>>>>>>>;
	// create a parametric network
	net_type2 pnet(dlib::prelu_(0.2), dlib::prelu_(0.25),
		dlib::repeat_group(dlib::prelu_(0.3), dlib::prelu_(0.4)));

	// print the detail of the pnet
	std::cout << "The pnet has: " << pnet.num_layers << " layers in it:\n";
	std::cout << pnet << std::endl;


	std::system("pause");
    return 0;
}

