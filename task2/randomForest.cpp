#include "randomForest.hpp"

/*randomForest::randomForest();
randomForest::~randomForest();*/
void RandomForest::creat(int nrTrees, int CVFoolds, int maxCategories, int maxDepth, int minSampelCount){
	for(int i = 0; i < nrTrees; i++){
		Ptr<ml::DTrees> dTree = ml::DTrees::create();
		dTree->setMaxDepth(maxDepth);
		dTree->setMinSampleCount(minSampelCount);
		dTree->setMaxCategories(maxCategories);
		dTree->setCVFolds(CVFoolds);
		randomForest.push_back(dTree);
	}
}
/*void RandomForest::train(Mat features, Mat label);
std::vector<int> RandForest::predict(Mat descriptorPicture);*/
int RandomForest::getNrTrees(){
	return randomForest.size();
}