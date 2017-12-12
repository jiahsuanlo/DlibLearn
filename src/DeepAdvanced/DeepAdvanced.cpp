// dlib dnn practice

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

using namespace dlib; // conv layer

// define a resnet block layer
template<int num_filter,
	template<typename> class BN,
	int stride,
	typename SUBNET>
	using block = BN < con<num_filter, 3, 3, 1, 1,
	relu<BN<con<num_filter, 3, 3, stride, stride,
	SUBNET>>>>>;

// define a resnet residual block
template<
	template<int, template<typename> class, int, typename> class block,
	int num_filter,
	template<typename> class BN,
	typename SUBNET>
	using residual = add_prev1<block<num_filter, BN, 1, tag1<SUBNET>>>;

// define downsampling (stride2) residual block
template<
	template<int, template<typename> class, int, typename> class block,
	int num_filter,
	template<typename> class BN,
	typename SUBNET>
	using residual_down = add_prev2 < avg_pool<2, 2, 2, 2,
	skip1<tag2<block<num_filter, BN, 2, tag1<SUBNET>>>>>>;

// now define 4 different residual blocks
template <typename SUBNET> using res = relu<residual<block, 8, bn_con, SUBNET>>;
template <typename SUBNET> using res_a = relu < residual<block, 8, affine, SUBNET>>;
template <typename  SUBNET> using res_down = relu<residual_down<block, 8, bn_con, SUBNET>>;
template <typename SUBNET> using resa_down = relu<residual_down<block, 8, affine, SUBNET>>;

// building the net type
const unsigned long num_classes = 10;
using net_type = loss_multiclass_log < fc<num_classes,
	avg_pool_everything<res<res<res<res_down<
	repeat<9, res,
	res_down<res<
	input<matrix<unsigned char>>>>>>>>>>>>;


int main(int argc, char**argv)
{
	// load data
	std::vector<matrix<unsigned char>> training_images;
	std::vector<unsigned long> training_labels;
	std::vector<matrix<unsigned char>> testing_images;
	std::vector<unsigned long> testing_labels;
	load_mnist_dataset("../data/mnistImages", training_images, training_labels, testing_images, testing_labels);

	// use smaller ram 
	set_dnn_prefer_smallest_algorithms();

	// create a net
	net_type net;

	// print layer info
	std::cout << "net has " << net.num_layers << " layers\n";
	//std::cout << net << "\n";
	
	// get output for layer 3 ==>  layer<3>(net).get_output()
	
	// now set trainer
	dnn_trainer<net_type, adam> trainer(net, adam(0.0005, 0.9, 0.999));
	trainer.be_verbose();
	trainer.set_iterations_without_progress_threshold(2000);
	trainer.set_learning_rate_shrink_factor(0.1);
	trainer.set_learning_rate(0.001);
	trainer.set_synchronization_file("./DeepAdvanced.dir/mnist_res_sync",
		std::chrono::seconds(100));

	// set mini batch
	std::vector<matrix<unsigned char>> mini_batch_samples;
	std::vector<unsigned long> mini_batch_labels;

	while (trainer.get_learning_rate >= 1e-6)
	{

	}
	
	std::system("pause");
	return 0;
}

