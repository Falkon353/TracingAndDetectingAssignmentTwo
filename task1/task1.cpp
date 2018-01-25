#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "../hog_visualization.cpp"
#include "iostream"



using namespace cv;
using namespace std;

int main(){
	Mat image;
	image = imread("./obj1000.jpg",CV_LOAD_IMAGE_COLOR);
	if(!image.data){
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	// resize and show resized immage
	resize(image, image, Size(128,96));
	namedWindow("Display windo", CV_WINDOW_AUTOSIZE);
	imshow("Display windo", image);

	// creat and show diferent variations of the image.
	// Gray scale 
	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);
	namedWindow("gray windo", CV_WINDOW_AUTOSIZE);
	imshow("gray windo", gray);

	// rotated
	Mat rotate;
	cv::rotate(image,rotate,ROTATE_90_CLOCKWISE);
	namedWindow("Rotate windo", CV_WINDOW_AUTOSIZE);
	imshow("Rotate windo", rotate);

	//fliped
	Mat flip; 
	cv::flip(image, flip, 1);
	namedWindow("Flipe windo", CV_WINDOW_AUTOSIZE);
	imshow("Flipe windo", flip);

	// init descriptors and HOG
	std::vector<float> descriptorsImage;
	std::vector<float> descriptorsGray;
	std::vector<float> descriptorsRotate;
	std::vector<float> descriptorsFlip;
	HOGDescriptor hog(Size(image.size().width,image.size().height), Size(16,12), Size(8,6), Size(16,12),9);
	//calculate and show HOG descriptors
	/*hog.compute(image,descriptorsImage);
	visualizeHOG(image,descriptorsImage,hog,5);*/

	hog.compute(gray,descriptorsGray);
	visualizeHOG(gray,descriptorsGray,hog,5);

	/*hog.compute(rotate,descriptorsRotate);
	visualizeHOG(rotate,descriptorsRotate,hog,5);*/

	/*hog.compute(flip,descriptorsFlip);
	visualizeHOG(flip,descriptorsFlip,hog,5);*/

	waitKey(0);
	return 0;
}