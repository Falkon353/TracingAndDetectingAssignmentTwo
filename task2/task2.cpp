#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"
#include "../hog_visualization.cpp"
#include "utilities.hpp"
#include "iostream"
#include "randomForest.hpp"



using namespace cv;
using namespace std;

/*int main(){
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
	dec_trees->setMaxDepth(10);
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
	HOGDescriptor hog(Size(image1.size().width,image1.size().height), Size(16,12), Size(8,6), Size(16,12),9);
	hog.compute(image1,descriptorImage1);
	hog.compute(image2,descriptorImage2);	
	Mat feats, labels;
	Mat mDescriptorImage1 = Mat(1, descriptorImage1.size(), CV_32FC1);
	memcpy(mDescriptorImage1.data, descriptorImage1.data(), descriptorImage1.size());
	Mat mDescriptorImage2 = Mat(1, descriptorImage2.size(), CV_32FC1);
	memcpy(mDescriptorImage2.data, descriptorImage2.data(), descriptorImage2.size()); 
	feats.push_back(mDescriptorImage1);
	feats.push_back(mDescriptorImage2);
	labels.push_back(-1);
	labels.push_back(1);
	cout << "Depth: " << dec_trees -> getMaxDepth() << std::endl;
	dec_trees -> train(ml::TrainData::create(feats,ml::ROW_SAMPLE, labels));
	cout << "Depth: " << dec_trees -> getMaxDepth() << std::endl;
	return 0;
}*/

/*int main(){
	std::vector< vector<string> > pathToPictures;
    string baseDir = "task2Data/train/";
    reqursivelyFindJPG(baseDir, pathToPictures);
    /*for(vector<string>& paths: pathToPictures){
        //cout << "Hello World!" << std::endl;
    	//cout << paths.size()  << std::endl;
      	for(string& picturPath: paths){
        	cout << "Picture: " << picturPath << std::endl;
      	}
	}*/
/*	cout << pathToPictures.size() << std::endl;
    Mat features, labels;
    std::vector<int> vecLabels;
    createTrainingData(features,labels,pathToPictures,vecLabels);
    cout << "features size" << features.size() << std::endl;
    cout << "labels size" << labels.size() << std::endl;
    //creat tree
	Ptr<ml::DTrees> dec_trees = ml::DTrees::create();
	dec_trees->setMaxDepth(20);
	dec_trees->setMinSampleCount(2);
	dec_trees->setRegressionAccuracy(0.01f);
	dec_trees->setUseSurrogates(false);
	dec_trees->setMaxCategories(6);
	dec_trees->setCVFolds(1);
	dec_trees->setUse1SERule(true);
	dec_trees->setTruncatePrunedTree(true);
	//dec_trees->setPriors(noArray());
	dec_trees -> train(ml::TrainData::create(features,ml::ROW_SAMPLE, labels));
	HOGDescriptor hog(Size(128,96), Size(16,12), Size(8,6), Size(16,12),9);
	std::vector< vector<string> > pathToTestPictures;
    string baseDirForTest = "task2Data/test/";
    reqursivelyFindJPG(baseDirForTest, pathToTestPictures);
    std::vector<vector<int> > answers;
    /*for(vector<string>& paths: pathToTestPictures){
        //cout << "Hello World!" << std::endl;
    	//cout << paths.size()  << std::endl;
      	for(string& picturPath: paths){
        	cout << "Picture: " << picturPath << std::endl;
      	}
	}*/
/*	dTreePredict(hog,dec_trees,pathToTestPictures,answers);
	for(vector<int>& folderAnswer: answers){
		cout << "New folder" << std::endl;
      	for(int& answer: folderAnswer){
        	cout << "Answer: " << answer << std::endl;
      	}
	}
    return 0;
}*/

int main(){
	RandomForest randForest;
	randForest.creat(50,1,6,10,2);
	cout << "size: " << randForest.getNrTrees() << std::endl;
	std::vector< vector<string> > pathToPictures;
    string baseDir = "task2Data/train/";
    reqursivelyFindJPG(baseDir, pathToPictures);
	Mat features, labels;
	std::vector<int> vecLabels;
    createTrainingData(features,labels,pathToPictures,vecLabels);
    randForest.train(features,labels,vecLabels);
    cout << "vecLabels size: " << vecLabels.size() << std::endl;
    cout << "vecLabels element 0: " << vecLabels[0] << std::endl;
    Mat mDescriptorPicture;
    /*picture = imread("task2Data/test/00/0049.jpg",CV_LOAD_IMAGE_COLOR);
    if(!picture.data){
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    resize(picture, picture, Size(128,96));*/
    //std::vector<float> descriptorPicture;
    string picturePath = "task2Data/test";
    HOGDescriptor hog(Size(128,96), Size(16,12), Size(8,6), Size(16,12),9);
    /*hog.compute(picture,descriptorPicture);
    Mat mDescriptorPicture = Mat(1, descriptorPicture.size(), CV_32FC1);
    memcpy(mDescriptorPicture.data, descriptorPicture.data(), descriptorPicture.size());*/
    creatPicturDecriptor(mDescriptorPicture,hog,picturePath);
    std::vector<std::vector<pair<int, int> > > answers;
    randForest.predictFromPath(hog,picturePath,vecLabels,answers);
    //randForest.predict(mDescriptorPicture,vecLabels,answers);
    for(std::vector<pair<int, int> > folder: answers){
    	cout << "NewFolder: " << std::endl;
    	for (pair<int, int> answer: folder){
    		cout << "Answer: " << answer.first << " Prosent: " << answer.second << std::endl;
    	}
    }
	return 0;
}

/*int main(){
	Mat mDescriptorPicture;
	string picturePath = "task2Data/train/00/0000.jpg";
	HOGDescriptor hog(Size(128,96), Size(16,12), Size(8,6), Size(16,12),9);
	int temp = creatPicturDecriptor(mDescriptorPicture,hog,picturePath);
	cout << "mDescriptorPicture size: " << mDescriptorPicture.size << std::endl;
	return 0;
}*/
