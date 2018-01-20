#ifndef RandomForest_H
#define RandomForest_H
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;

class RandomForest{
	std::vector<ml::DTrees> randomForest;

public:
	/*randomForest();
	~randomForest();*/
	void creat(int nrTrees, int CVFoolds, int maxCategories, int maxDepth, int minSampelCount);
	void train(Mat features, Mat label);
	std::vector<int> predict(Mat descriptorPicture);
	int getNrTrees(); 
	
};

/*randomForest::randomForest();
randomForest::~randomForest();*/
/*void RandomForest::creat(int nrTrees, int CVFoolds, int maxCategories, int maxDepth, int minSampelCount);
void RandomForest::train(Mat features, Mat label);
std::vector<int> RandomForest::predict(Mat descriptorPicture);
int RandomForest::getNrTrees();*/

#endif