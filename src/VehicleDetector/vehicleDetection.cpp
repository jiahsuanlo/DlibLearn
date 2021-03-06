// VehicleDetection.cpp : Defines the entry point for the console application.
//

#include <iostream> 
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <ctime>

using namespace dlib;

// The rear view vehicle detector network
template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;
template <typename SUBNET> using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = relu<affine<con5<55, SUBNET>>>;
using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

int main()
{
	net_type net;
	dlib::shape_predictor sp;

	// load weights
	dlib::deserialize("../mmod_rear_end_vehicle_detector.dat") >> net >> sp;
	// load image
	dlib::matrix<dlib::rgb_pixel> img;
	dlib::load_image(img, "../vehicle.jpg");

	// create window
	dlib::image_window win;
	win.set_image(img);

	// run the detector on the image
	auto t0 = std::clock();
	for (auto&& d : net(img))
	{
		auto fd = sp(img, d);
		dlib::rectangle rect;
		for (unsigned long j = 0; j < fd.num_parts(); ++j)
			rect += fd.part(j);
		win.add_overlay(rect, rgb_pixel(255, 0, 0));
	}
	std::cout << "elapsed time= " << double(std::clock() - t0) / CLOCKS_PER_SEC / 100. << "\n";

	// step-wise
	std::cout << "Hit enter to see the intermediate steps...\n";
	std::cin.get();

	// pyramid settings
	const float lower = -2.5;
	const float upper = 0.0;
	std::cout << "Jet color mapping range: lower= " << lower << ", upper= " << upper << "\n";

	// create a tiled pyramid image
	std::vector<dlib::rectangle> rects;
	dlib::matrix<dlib::rgb_pixel> tiled_img;
	using pyramid_type = std::remove_reference<decltype(dlib::input_layer(net))>::type::pyramid_type;
	dlib::create_tiled_pyramid<pyramid_type>(img, tiled_img, rects,
		dlib::input_layer(net).get_pyramid_padding(),
		dlib::input_layer(net).get_pyramid_outer_padding());
	dlib::image_window winpyr(tiled_img, "Tiled Pyramid");

	// show sliding window detector
	std::cout << "Number of channels in final tensor image: " <<
		net.subnet().get_output().k() << "\n";
	dlib::matrix<float> network_output = dlib::image_plane(net.subnet().get_output(), 0, 0);
	for (long k = 1; k < net.subnet().get_output().k(); k++)
	{
		network_output = dlib::max_pointwise(network_output,
			dlib::image_plane(net.subnet().get_output(), 0, 1));
	}

	// upsampling the output image, since the CNN has 8x downsampling
	const double network_output_scale = img.nc() / (double)network_output.nc();
	dlib::resize_image(network_output_scale, network_output);

	// display output image as colored one
	dlib::image_window win_output(dlib::jet(network_output, upper, lower),
		"Output tensor from the CNN");

	// overlay the network output on the pyramid images
	for (long r = 0; r < tiled_img.nr(); ++r)
	{
		for (long c = 0; c < tiled_img.nc(); ++c)
		{
			dlib::dpoint tmp(c, r);
			// get output 
			tmp = dlib::input_tensor_to_output_tensor(net, tmp);
			// scale up
			tmp = dlib::point(network_output_scale*tmp);

			if (dlib::get_rect(network_output).contains(tmp))
			{
				// alpha blending
				float val = network_output(tmp.y(), tmp.x());
				dlib::rgb_alpha_pixel p;
				dlib::assign_pixel(p, colormap_jet(val, lower, upper));
				p.alpha = 120;
				dlib::assign_pixel(tiled_img(r, c), p);
			}
		}
	}
	dlib::image_window win_pyr_overlay(tiled_img, "Detection on image pyramid");

	// collapse pyramid images back to the original image
	dlib::matrix<float> collapsed(img.nr(), img.nc());
	dlib::resizable_tensor input_tensor;
	dlib::input_layer(net).to_tensor(&img, &img + 1, input_tensor); //??

	for (long r = 0; r < collapsed.nr(); r++)
	{
		for (long c = 0; c < collapsed.nc(); c++)
		{
			// loop over scales and find the max score
			float max_score = -1e30;
			for (double scale = 1; scale > 0.2; scale *= 5.0 / 6.0)
			{
				// image to tiled pyramid
				dpoint tmp = dlib::center(dlib::input_layer(net).
					image_space_to_tensor_space(input_tensor, scale, dlib::drectangle(dpoint(c, r))));
				// map the pyramid to output coords
				tmp = dlib::point(network_output_scale*dlib::input_tensor_to_output_tensor(net, tmp));

				if (dlib::get_rect(network_output).contains(tmp))
				{
					float val = network_output(tmp.y(), tmp.x());
					if (val > max_score)
					{
						max_score = val;
					}
				}
			}
			collapsed(r, c) = max_score;
			// blend overlay
			dlib::rgb_alpha_pixel p;
			dlib::assign_pixel(p, dlib::colormap_jet(max_score, lower, upper));
			p.alpha = 120;
			dlib::assign_pixel(img(r, c), p);
		}
	}
	dlib::image_window win_collapsed(dlib::jet(img, lower, upper), "Collapsed tensor from the network");
	dlib::image_window win_img_and_sal(img, "collapsed detection score");

	std::cout << "Hit enter to stop program...\n";
	std::cin.get();

	return 0;
}

