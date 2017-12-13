#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace dlib;

/* ------ Define the CNN for face detection -------------
 target input image is 50 x 50
 3 downsampling layers (8x reduction)
 4 conv layers
*/

// 5x5 conv with 2x downsampling block
template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
// 3x3 conv no downsampling block
template <long num_filters, typename SUBNET> using con3 = con<num_filters, 3, 3, 1, 1, SUBNET>;
// 8x downsampling block using 3 5x5 conv, 32 channels
template <typename SUBNET> using downsampler = relu<bn_con<con5d<32,
	relu<bn_con<con5d<32,
	relu<bn_con<con5d<32,SUBNET>>>>>>>>>;
// rest of the network: 3x3 conv with batch normalization
template <typename SUBNET> using rcon3 = relu<bn_con<con3<32, SUBNET>>>;

// finally add one-channel, 6x6 classifier layer and loss_mmod to complete the network
using net_type = loss_mmod<con<1, 6, 6, 1, 1,
	rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// --- main ----
int main(int argc, char** argv)
{
	const std::string face_dir = "../data/faces";

	// load dataset
	std::vector<matrix<rgb_pixel>> train_images, test_images;
	std::vector<std::vector<mmod_rect>> train_boxes, test_boxes;
	
	load_image_dataset(train_images, train_boxes, face_dir+"/training.xml");
	load_image_dataset(test_images, test_boxes, face_dir+"/testing.xml");

	std::cout << "num train images= " << train_images.size() << "\n";
	std::cout << "num of test imamges= " << test_images.size() << "\n";

	// set mmod options- pick the minimal size of face in pixels
	mmod_options options(train_boxes, 40, 40);
	std::cout << "num detector windows" << options.detector_windows.size() << "\n";
	for (auto w : options.detector_windows)
		std::cout << "detector window width by height: " << w.width << " x " << w.height << "\n";
	std::cout << "overlap NMS IOU thresh: " << options.overlaps_nms.get_iou_thresh() << "\n";
	std::cout << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << "\n";

	// create the net
	net_type net(options);
	// The MMOD loss requires that the number of filters in the final network layer equal
	// options.detector_windows.size().  So we set that here as well.
	net.subnet().layer_details().set_num_filters(options.detector_windows.size());
	dnn_trainer<net_type> trainer(net);


	

	std::system("pause");
	return 0;
}



