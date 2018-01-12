#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "hog_visualization.cpp"
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
	Mat gray;

	cvtColor(image, gray, CV_BGR2GRAY);

	namedWindow("Display windo", CV_WINDOW_AUTOSIZE);
	imshow("Display windo", image);

	Mat rotate;

	cv::rotate(image,rotate,ROTATE_90_CLOCKWISE);

	namedWindow("Rotate windo", CV_WINDOW_AUTOSIZE);
	imshow("Rotate windo", rotate);

	Mat flip; 

	cv::flip(image, flip, 1);

	namedWindow("Flipe windo", CV_WINDOW_AUTOSIZE);
	imshow("Flipe windo", flip);

	try{
		imwrite("test.jpg", gray);
	}
	catch (runtime_error& ex){
		cout << "Was not able to store image" << std::endl;
		return 1;
	}

	cout << "Was able to stor image in test.jpg" << std::endl;



	namedWindow("Result windo", CV_WINDOW_AUTOSIZE);
	imshow("Result windo", gray);

	std::vector<float> descriptors;
	std::vector<Point> points;
	Point testPoint(-240,-340);
	Point testPointTwo(-500,-1000);
	//points.push_back(testPoint);
	points.push_back(testPointTwo);
	//cout << "hei" << std::endl;
	HOGDescriptor desc(Size(64,128), Size(16,8), Size(8,8), Size(8,8),9);	
	desc.compute(image,descriptors, Size(64,128),Size(8,8),points);
	visualizeHOG(image,descriptors,desc,5);

	waitKey(0);
	return 0;
}