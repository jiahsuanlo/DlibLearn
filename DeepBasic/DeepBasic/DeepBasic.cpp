// DeepBasic.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

int main()
{
	// load the data 
	std::vector<dlib::matrix<unsigned char>> train_images;
	std::vector<unsigned long> train_labels;
	std::vector<dlib::matrix<unsigned char>> test_images;
	std::vector<unsigned long> test_labels;
	dlib::load_mnist_dataset("./mnistImages", train_images, train_labels,
		test_images, test_labels);

	// define leNet
	using net_type = dlib::loss_multiclass_log <
		dlib::fc < 10,   // fully-connected 10 output
		dlib::relu < dlib::fc < 84,  // fully-connected 84 output -> relu
		dlib::relu < dlib::fc < 120,  // fully-connected 120 output->relu
		dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<16, 5, 5, 1, 1, // convolute 5x5x16 ->relu-> maxpool
		dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<6, 5, 5, 1, 1, // convolute 5x5x6 -> relu->maxpool
		dlib::input<dlib::matrix<unsigned char>>  // input 28x28 image
		>>>>>>>>>>>>;
	net_type net;

	// train the net
	dlib::dnn_trainer<net_type> trainer(net);
	trainer.set_learning_rate(0.01);
	trainer.set_min_learning_rate(0.00001);
	trainer.set_mini_batch_size(128);
	trainer.be_verbose();

	// save results every 20 second
	trainer.set_synchronization_file("mnist_sync", std::chrono::seconds(20));

	// start training
	trainer.train(train_images, train_labels);

	// save the trained model
	net.clean();
	dlib::serialize("mnist_network.dat") << net;






    return 0;
}

