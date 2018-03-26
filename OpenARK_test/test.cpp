// C++ Libraries
#include <stdio.h>
#include <string>
#include <time.h>

// OpenCV Libraries
#include <opencv/cxcore.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

// OpenARK Libraries
#include "../include/core.h"
#include "../include/DepthCamera.h"
#include "../include/SR300Camera.h"
#include "../include/Visualizer.h"

// dlib
//#include "dlib/all/source.cpp"
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/gui_widgets.h>

using namespace cv;
using namespace dlib;
using namespace std;
using namespace ark;

int main() {
	// initialize
	DepthCamera::Ptr camera = std::make_shared<SR300Camera>();
	DetectionParams::Ptr params = DetectionParams::create();

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

	camera->beginCapture();

	int frame = 0;
	while (true)
	{
		// // Show image
		cv::Mat xyzMap = camera->getXYZMap();
		cv::Mat irMap = camera->getIRMap(); 

		cv::Mat grayVisual; cv::cvtColor(camera->getIRMap(), grayVisual, cv::COLOR_GRAY2BGR, 3);
		cv::Mat xyzVisual; ark::Visualizer::visualizeXYZMap(camera->getXYZMap(), xyzVisual);
		cv::imshow("XYZ Map", xyzVisual);
		// cv::imshow("Result", grayVisual);

		/**** Start: Loop Break Condition ****/
		int c = cv::waitKey(1);
		if (c == 'q' || c == 'Q' || c == 27) {
			break;
		} else if (c == 'w' || c == 'W' || c == 33) {
			// Detect faces 
			cv_image<bgr_pixel> cimg(grayVisual);
			std::vector<dlib::rectangle> faces = detector(cimg);
			// Find the pose of each face.
			std::vector<full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i) {
				full_object_detection shape = pose_model(cimg, faces[i]);
				point p1 = shape.part(0) / 2 + shape.part(1) / 2;
				point p2 = shape.part(16) / 2 + shape.part(15) / 2;

				cv::Vec3f * ptr1 = xyzMap.ptr<cv::Vec3f>(p1.x());
				cv::Vec3f xyz1 = ptr1[p1.y()];
				cout << "point 1: " << xyz1 << endl;

				cv::Vec3f * ptr2 = xyzMap.ptr<cv::Vec3f>(p2.x());
				cv::Vec3f xyz2 = ptr1[p2.y()];
				cout << "point 2: " << xyz2 << endl;

				cv::Vec3f d_vec = xyz1 - xyz2;
				float dist = sqrt(d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] * d_vec[2] * d_vec[2]);
				cout << "distance: " << dist << endl;

				draw_line(cimg, shape.part(0) / 2 + shape.part(1) / 2, shape.part(16) / 2 + shape.part(15) / 2, 1);
				shapes.push_back(shape);
			}

			cv::imshow("Result", grayVisual);
			while (true) {
				int d = cv::waitKey(1);
				if (d == 'q' || d == 'Q' || d == 27) {
					break;
				}
			}
		}
		/**** End: Loop Break Condition ****/

		// Turn OpenCV's Mat into something dlib can deal with.  Note that this just
		// wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
		// long as temp is valid.  Also don't do anything to temp that would cause it
		// to reallocate the memory which stores the image as that will make cimg
		// contain dangling pointers.  This basically means you shouldn't modify temp
		// while using cimg.

		frame++;
	}

    camera->endCapture();

    cv::destroyAllWindows();
    return 0;
}