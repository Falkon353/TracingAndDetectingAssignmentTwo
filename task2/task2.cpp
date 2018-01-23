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


int main(){
	//creat training data
	string baseDir = "task2Data/train/";
	std::vector< vector<string> > pathToPicturesTraining;
    reqursivelyFindJPG(baseDir, pathToPicturesTraining);
	Mat features, labels;
	std::vector<int> vecLabels;
    createTrainingData(features,labels,pathToPicturesTraining,vecLabels);
    //Creat and train DTree
    Ptr<ml::DTrees> dec_trees = ml::DTrees::create();
	dec_trees->setMaxDepth(20);
	dec_trees->setMinSampleCount(2);
	dec_trees->setMaxCategories(6);
	dec_trees->setCVFolds(1);
	dec_trees->setUse1SERule(true);
	dec_trees->setTruncatePrunedTree(true);
	dec_trees -> train(ml::TrainData::create(features,ml::ROW_SAMPLE, labels));
    //Creat and train random forest
	RandomForest randForest;
	randForest.creat(50,1,6,20,2);
    randForest.train(features,labels,vecLabels);
    //Use DTree to predict.
    HOGDescriptor hog(Size(128,96), Size(16,12), Size(8,6), Size(16,12),9);
    string baseDirTest = "task2Data/test/";
    std::vector<vector <string> > pathToPicturesTest;
    reqursivelyFindJPG(baseDirTest,pathToPicturesTest);
    std::vector<vector<int> > answersDTree; 
    dTreePredict(hog,dec_trees,pathToPicturesTest,answersDTree);
	for(vector<int>& folderAnswer: answersDTree){
		cout << "New folder" << std::endl;
      	for(int& answer: folderAnswer){
        	cout << "Answer DTree: " << answer << std::endl;
      	}
	}
    //Use random forest to predict.
    Mat mDescriptorPicture;
    std::vector<std::vector<pair<int, int> > > answersRandomForest;
    randForest.predictFromPath(hog,baseDirTest,vecLabels,answersRandomForest);
    for(std::vector<pair<int, int> > folder: answersRandomForest){
    	cout << "NewFolder: " << std::endl;
    	for (pair<int, int> answer: folder){
    		cout << "Answer RandomForest: " << answer.first << " Prosent: " << answer.second << std::endl;
    	}
    }
	return 0;
}

