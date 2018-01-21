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
void RandomForest::train(Mat features, Mat labels, std::vector<int> vecLabels){
	this-> vecLabels = vecLabels;
	srand (time(NULL));
	Mat feturesSubset, labelsSubset;
	int nrTotalfeatures = features.rows;
	cout << "nrTotalfeatures: " << nrTotalfeatures << std::endl;
	cout << "size features: " << features.size() << std::endl;
	int nrFeatursPerTree = nrTotalfeatures/randomForest.size();
	std::vector<int> usedFeaturs;
	int randFeatur, nrNewFeaturs;
	for(Ptr<ml::DTrees> dTree: randomForest){
		feturesSubset.release();
		labelsSubset.release();
		nrNewFeaturs = 0;
		while(nrNewFeaturs < nrFeatursPerTree){
			randFeatur = rand() % nrTotalfeatures;
			//cout << "randFeatur: " << randFeatur << std::endl;
			if ( std::find(usedFeaturs.begin(), usedFeaturs.end(), randFeatur) != usedFeaturs.end() ){
				continue;
			}
			nrNewFeaturs++;
			feturesSubset.push_back(features(Range(randFeatur,randFeatur+1),Range::all()));
			labelsSubset.push_back(labels(Range(randFeatur,randFeatur+1),Range::all()));
		}
		cout << "feturesSubset size: " << feturesSubset.size() << std::endl;
		cout << "labelsSubset size: " << labelsSubset.size() << std::endl;
		dTree-> train(ml::TrainData::create(feturesSubset,ml::ROW_SAMPLE, labelsSubset));
	}
}
void RandomForest::predict(Mat descriptorPicture,std::vector<int> vecLabels, std::vector<int>& answer){
	std::vector<int> answers;
	for(Ptr<ml::DTrees> dTree: randomForest){
		int answer = dTree -> predict(descriptorPicture);
		cout << "answer: " << answer << std::endl;
		answers.push_back(answer);
	}
	int mostVotedLabe = 0;
	int votsMostVotedLabe = 0;
	int labelVots;
	cout << "foo" << std::endl;
	for(int label: vecLabels){
		labelVots = count (answers.begin(), answers.end(), label);
		//cout << "labelVots: " << labelVots << std::endl;
		if(labelVots > votsMostVotedLabe){
			mostVotedLabe = label;
			votsMostVotedLabe = labelVots;
			cout << "mostVotedLabe: " << mostVotedLabe << "votsMostVotedLabe: " << votsMostVotedLabe << std::endl;
		}
	}
	cout << "mostVotedLabe: " << mostVotedLabe << std::endl;
	answer.push_back(mostVotedLabe);
	//answers.push_back((votsMostVotedLabe*100)/vecLabels.size());
}
int RandomForest::getNrTrees(){
	return randomForest.size();
}