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
	int nrFeatursPerTree = nrTotalfeatures/10;
	std::vector<int> usedFeaturs, globalUsedFeaturs;
	int randFeatur, nrNewFeaturs;
	for(Ptr<ml::DTrees> dTree: randomForest){
		//int nrFeatursPerTree = rand() % nrTotalfeatures;
		int uniqFeaturNr = 0;
		feturesSubset.release();
		labelsSubset.release();
		nrNewFeaturs = 0; 
		while(nrNewFeaturs < nrFeatursPerTree){
			randFeatur = rand() % nrTotalfeatures;
			//cout << "randFeatur: " << randFeatur << std::endl;
			/*if ( std::find(usedFeaturs.begin(), usedFeaturs.end(), randFeatur) != usedFeaturs.end() ){
				continue;
			} else*//* if(std::find(globalUsedFeaturs.begin(), globalUsedFeaturs.end(), randFeatur) != globalUsedFeaturs.end() && uniqFeaturNr < nrTotalfeatures/randomForest.size()){
				//cout << "inside first elif" << globalUsedFeaturs.size() << std::endl;
				continue;
			} else if(!(std::find(globalUsedFeaturs.begin(), globalUsedFeaturs.end(), randFeatur) != globalUsedFeaturs.end()) && uniqFeaturNr < nrTotalfeatures/randomForest.size()){
				//cout << "inside second elf" << std::endl;
				uniqFeaturNr++;
				globalUsedFeaturs.push_back(randFeatur);
			}*/
			nrNewFeaturs++;
			feturesSubset.push_back(features(Range(randFeatur,randFeatur+1),Range::all()));
			labelsSubset.push_back(labels(Range(randFeatur,randFeatur+1),Range::all()));
			usedFeaturs.push_back(randFeatur);
		}
		//cout << "feturesSubset size: " << feturesSubset.size() << std::endl;
		//cout << "labelsSubset size: " << labelsSubset.size() << std::endl;

		usedFeaturs.clear();
		//cout << "feturesSubset size: " << feturesSubset.size() << std::endl;
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
		//cout << "label: " << label << " Vots: " << labelVots << std::endl;
		if(labelVots > votsMostVotedLabe){
			mostVotedLabe = label;
			votsMostVotedLabe = labelVots;
			//cout << "mostVotedLabe: " << mostVotedLabe << "votsMostVotedLabe: " << votsMostVotedLabe << std::endl;
		}
	}
	//cout << "mostVotedLabe: " << mostVotedLabe << std::endl;
	answer.first = mostVotedLabe;
	answer.second = (votsMostVotedLabe*100)/randomForest.size();
}

void RandomForest::predictFromPath(HOGDescriptor& hog, string baseDir, std::vector<int> vecLabels, std::vector<std::vector<pair<int, int> > >& answers){
	std::vector<std::vector<string> > pathToPictures;
	reqursivelyFindJPG(baseDir,pathToPictures);
	for(std::vector<string> paths: pathToPictures){
		std::vector<pair<int, int> > folderAnswers;
		for(string pathToPicture: paths){
			pair<int, int> picturAnswer;
			vector<Mat> vectorDescriptorsPicture;
			creatPicturDecriptor(vectorDescriptorsPicture,hog, pathToPicture);
			for(Mat mDescriptorPicture: vectorDescriptorsPicture){
				cout << "DescriptorPicture: " << mDescriptorPicture.size() << std::endl;
	            this->predict(mDescriptorPicture,vecLabels,picturAnswer);
	            //cout << "pathToPictures" << pathToPicture << std::endl;
	            //cout << "answer: " << picturAnswer.first << " Prosent: " << picturAnswer.second << std::endl;
	            folderAnswers.push_back(picturAnswer);
			}
		}
		answers.push_back(folderAnswers);
	}
}

void RandomForest::evaluateBoxes(Mat& picture, HOGDescriptor& hog, std::vector<vector<int> > boxes, std::vector<int> vecLabels, std::vector<pair<vector<int>, pair<int, int> > >& boxPredictionPairs){
	for(std::vector<int> boxCordinats: boxes){
        pair<int,int> answer;
        //HOGDescriptor hog(Size(box.cols,box.rows), Size(box.cols/2,box.rows/2), Size(box.cols/4,box.rows/4), Size(box.cols/2,box.cols/2),9);
        //HOGDescriptor hog(Size(,12), Size(8,6), Size(4,3), Size(8,6),9);
        Mat box = picture(Range(boxCordinats[0],boxCordinats[1]+1),Range(boxCordinats[2],boxCordinats[3]+1));
        resize(box, box, Size(PICTURE_COL,PICTURE_ROW));
        std::vector<float> descriptorPicture;
        hog.compute(box,descriptorPicture);
        Mat mDescriptorPicture;
        mDescriptorPicture = Mat(1, descriptorPicture.size(), CV_32FC1);
    	memcpy(mDescriptorPicture.data, descriptorPicture.data(), descriptorPicture.size());
    	//cout << "box featur size: " << mDescriptorPicture.size() << std::endl;
    	cout << "DescriptorPicture: " << mDescriptorPicture.size() << std::endl;
        this->predict(mDescriptorPicture,vecLabels,answer);
        pair<vector<int>, pair<int,int> > boxPredictionPair;
        boxPredictionPair.first = boxCordinats;
        boxPredictionPair.second = answer;
        boxPredictionPairs.push_back(boxPredictionPair);
    }
}

int RandomForest::getNrTrees(){
	return randomForest.size();
}