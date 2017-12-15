/* Use dlib pre-trained model to detect vehicle from
front and back side views
*/

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>

using namespace dlib;

// The vehicle detector network
template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;
template <typename SUBNET> using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = relu<affine<con5<55, SUBNET>>>;
using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

int main(int argc, char** argv)
{
	// declare net
	net_type net;
	shape_predictor sp;

	// load pre-trained model from :http://dlib.net/files/mmod_front_and_rear_end_vehicle_detector.dat.bz2
	deserialize("../src/VehicleFrontBackDetector/mmod_front_and_rear_end_vehicle_detector.dat")>>
		net>>sp;

	// load vehicle test image
	matrix<rgb_pixel> img;
	load_image(img, "../data/vehicle_photos/0016E5_05550.png");

	image_window win;
	win.set_image(img);

	// run the detector
	for (auto&& d : net(img))
	{
		// use shape predictor to output four corners
		auto fourCorners = sp(img, d);
		rectangle rect;
		for (unsigned long j = 0; j < fourCorners.num_parts(); ++j)
		{
			rect += fourCorners.part(j);
		}
		if (d.label == "rear")
		{
			win.add_overlay(rect, rgb_pixel(255, 0, 0), d.label);
		}
		else
		{
			win.add_overlay(rect, rgb_pixel(0, 0, 255), d.label);
		}
	}


	std::system("pause");
	return 0;
}