// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image.  In
    particular, this program shows how you can take a list of images from the
    command line and display each on the screen with red boxes overlaid on each
    human face.

    The examples/faces folder contains some jpg images of people.  You can run
    this program on them and see the detections by executing the following command:
        ./face_detection_ex faces/*.jpg

    
    This face detector is made using the now classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  This type of object detector is fairly
    general and capable of detecting many types of semi-rigid objects in
    addition to human faces.  Therefore, if you are interested in making your
    own object detectors then read the fhog_object_detector_ex.cpp example
    program.  It shows how to use the machine learning tools which were used to
    create dlib's face detector. 


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/
//#define DLIB_JPEG_SUPPORT


#include "dlib/gui_widgets.h"
#include "dlib/image_io.h"
#include "dlib/image_processing.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/opencv.h"
#include <iostream>
#include <dirent.h>
#include <string>
#include <opencv2/opencv.hpp>






using namespace dlib;
using namespace std;


void get_image_names(std::string file_path, std::vector<std::string>& file_names)
{
    DIR *dir;
    struct dirent *ptr;
    dir = opendir(file_path.c_str());
    while( (ptr = readdir(dir)) != NULL)
    {
        string filename = string(ptr->d_name);
        if (filename == "." || filename == ".."){
            continue;
        }
        string path = file_path + string("/") + filename;
        file_names.push_back(path);
    }
    closedir(dir);
    sort(file_names.begin(), file_names.end());
}


int main(int argc, char** argv)
{  
	std::vector<string> file_names;
	//get_image_names(argv[1], file_names);
	cv::Mat img, img_gray;
	double t, s, fps;
	int count = 0;
	cv::VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	
  if (!cap.isOpened()){
    std::cout << "Failed to open camera." << std::endl;
    return -1;
  }
	
    try
    {
        /*if (argc == 1)
        {
            cout << "Give some image files as arguments to this program." << endl;
            return 0;
        }*/

        frontal_face_detector detector = get_frontal_face_detector();
        image_window win;

        // Loop over all the images provided on the command line.
        s = (double)cv::getTickCount();
        while(1)
        {
			//ReadImgInside();
			t = (double)cv::getTickCount();
			
            cap >> img;            
            cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
            cv_image<unsigned char> dlibImgFrameGray(img_gray);
            //load_image(img, file_names[i]);
            // Make the image bigger by a factor of two.  This is useful since
            // the face detector looks for faces that are about 80 by 80 pixels
            // or larger.  Therefore, if you want to find faces that are smaller
            // than that then you need to upsample the image as we do here by
            // calling pyramid_up().  So this will allow it to detect faces that
            // are at least 40 by 40 pixels in size.  We could call pyramid_up()
            // again to find even smaller faces, but note that every time we
            // upsample the image we make the detector run slower since it must
            // process a larger image.
            //pyramid_up(dlibImgFrameGray);

            // Now tell the face detector to give us a list of bounding boxes
            // around all the faces it can find in the image.
            
                    
            std::vector<rectangle> dets = detector(dlibImgFrameGray);

            cout << "Number of faces detected: " << dets.size() << endl;
            // Now we show the image on the screen and the face detections as
            // red overlay boxes.
            win.clear_overlay();
            win.set_image(dlibImgFrameGray);
            win.add_overlay(dets, rgb_pixel(255,0,0));
            

            cv::waitKey(10);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	    fps = 1.0 / t;
	    cout<<"Time consumed: "<< t << "s" << "   FPS: "<< fps<<endl;
	    count ++;
        }
	    
        s = ((double)cv::getTickCount() - s) / cv::getTickFrequency();
	fps = 1.0 / s;
	cout<< "Average FPS: " << (count+1)/s <<endl;
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------
