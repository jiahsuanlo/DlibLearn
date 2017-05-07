// PtCloud.cpp : Defines the entry point for the console application.
//
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>
#include <cmath>

#include "stdafx.h"

int main()
{
    // make a point cloud
	std::vector<dlib::perspective_window::overlay_dot> points;
	dlib::rand rnd;
	for (double i = 0; i < 20; i += 0.001)
	{
		// point on a spiral
		dlib::vector<double> val(cos(i), sin(i), i / 4);

		// add random noise
		dlib::vector<double> temp(rnd.get_random_gaussian(),
			rnd.get_random_gaussian(),
			rnd.get_random_gaussian());
		val += temp / 20;

		// pick a color based on the distance along the spiral
		dlib::rgb_pixel color = dlib::colormap_jet(i, 0, 20);

		// add to overlay point
		points.push_back(dlib::perspective_window::overlay_dot(val, color));
	}

	// display the point cloud
	dlib::perspective_window win;
	win.set_title("Point Cloud");
	win.add_overlay(points);
	win.wait_until_closed();
	return 0;
}

