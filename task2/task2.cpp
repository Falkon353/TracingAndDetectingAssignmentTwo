#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"
#include "../hog_visualization.cpp"
#include "iostream"



using namespace cv;
using namespace std;

int main(){
	//loade trainingdata
	Mat image1;
	image1 = imread("./task2Data/train/00/0000.jpg",CV_LOAD_IMAGE_COLOR);
	if(!image1.data){
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	resize(image1, image1, Size(128,96));
	Mat image2;
	image2 = imread("./task2Data/train/01/0000.jpg",CV_LOAD_IMAGE_COLOR);
	if(!image2.data){
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	resize(image2, image2, Size(128,96));
	//creat tree
	Ptr<ml::DTrees> dec_trees = ml::DTrees::create();
	dec_trees->setMaxDepth(1);
	dec_trees->setMinSampleCount(2);
	dec_trees->setRegressionAccuracy(0.01f);
	dec_trees->setUseSurrogates(false);
	dec_trees->setMaxCategories(2);
	dec_trees->setCVFolds(1);
	dec_trees->setUse1SERule(true);
	dec_trees->setTruncatePrunedTree(true);
	//dec_trees->setPriors(noArray());

	//Train tree.
	std::vector<float> descriptorImage1;
	std::vector<float> descriptorImage2;
	std::vector<std::vector<float> > descriptorsImage;
	HOGDescriptor hog(Size(image1.size().width,image1.size().height), Size(16,12), Size(8,6), Size(16,12),9);
	hog.compute(image1,descriptorImage1);
	hog.compute(image2,descriptorImage2);
	descriptorsImage.push_back(descriptorImage1);
	descriptorsImage.push_back(descriptorImage2);
	cout << descriptorsImage.size() << std::endl;
	std::vector<float> responses;
	dec_trees -> train(descriptorsImage,ml::COL_SAMPLE,noArray());
	//cout << binarTree -> getMaxDepth() << std::endl;	

	return 0;
}