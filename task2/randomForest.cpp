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
	//cout << "nrTotalfeatures: " << nrTotalfeatures << std::endl;
	//cout << "size features: " << features.size() << std::endl;
	int nrFeatursPerTree = nrTotalfeatures/3;
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
			usedFeaturs.push_back(randFeatur);
		}
		//cout << "feturesSubset size: " << feturesSubset.size() << std::endl;
		//cout << "labelsSubset size: " << labelsSubset.size() << std::endl;
		usedFeaturs.clear();
		dTree-> train(ml::TrainData::create(feturesSubset,ml::ROW_SAMPLE, labelsSubset));
	}
}
void RandomForest::predict(Mat descriptorPicture,std::vector<int> vecLabels, pair<int, int>& answer){
	std::vector<int> answers;
	for(Ptr<ml::DTrees> dTree: randomForest){
		int answer = dTree -> predict(descriptorPicture);
		//cout << "answer: " << answer << std::endl;
		answers.push_back(answer);
	}
	int mostVotedLabe = 0;
	int votsMostVotedLabe = 0;
	int labelVots;
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
	answer.first = mostVotedLabe;
	answer.second = (votsMostVotedLabe*100)/randomForest.size();
}

void RandomForest::predictFromPath(HOGDescriptor& hog, string picturePath, std::vector<int> vecLabels, std::vector<std::vector<pair<int, int> > >& answer){
	std::vector<std::vector<string> > pathToPictures;
	reqursivelyFindJPG(picturePath, std::vector< vector <string> >& pathToPictures);
	//std::vector<std::vector<int> > answer;
	for(std::vector<string> paths: pathToPictures){
		std::vector<pair<int, int> > folderAnswers;
		for(string pathToPicture: paths){
			pair<int, int> picturAnswer;
			Mat mDescriptorPicture;
			creatPicturDecriptor(mDescriptorPicture,hog, pathToPictures);
            this->predict(mDescriptorPicture,vecLabels,picturAnswer);
            folderAnswers.push_back(picturAnswer);
		}
		answers.push_back(folderAnswers);
	}
}
int RandomForest::getNrTrees(){
	return randomForest.size();
}