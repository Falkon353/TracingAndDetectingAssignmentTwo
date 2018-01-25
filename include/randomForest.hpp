#ifndef RandomForest_H
#define RandomForest_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"
#include "utilities.hpp"
#include "consts.hpp"

using namespace std;
using namespace cv;

class RandomForest{
	std::vector<Ptr<ml::DTrees> > randomForest;
	std::vector<int> vecLabels;

public:
	/*randomForest();
	~randomForest();*/
	void creat(int nrTrees, int CVFoolds, int maxCategories, int maxDepth, int minSampelCount);
	void train(Mat features, Mat label, std::vector<int> vecLabels);
	void predict(string picturePath, HOGDescriptor& hog, std::vector<int> vecLabels, pair<int, int>& answer);
	void predictFromPath(HOGDescriptor& hog, string baseDir, std::vector<int> vecLabels, std::vector<std::vector<pair<int, int> > >& answers);
	void evaluateBoxes(Mat& picture, HOGDescriptor& hog, std::vector<vector<int> > boxes, std::vector<int> vecLabels, std::vector<pair<vector<int>, pair<int, int> > >& boxPredictionPairs);
	int getNrTrees(); 
	
};

/*randomForest::randomForest();
randomForest::~randomForest();*/
/*void RandomForest::creat(int nrTrees, int CVFoolds, int maxCategories, int maxDepth, int minSampelCount);
void RandomForest::train(Mat features, Mat label);
std::vector<int> RandomForest::predict(Mat descriptorPicture);
int RandomForest::getNrTrees();*/

#endif