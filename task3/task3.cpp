#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"
#include <string>
#include "consts.hpp"
#include "utilities.hpp"
#include "iostream"
#include "randomForest.hpp"

int main(){
	//Creat Training data
	string baseDir = "task3Data/train/";
	std::vector< vector<string> > pathToPicturesTraining;
    reqursivelyFindJPG(baseDir, pathToPicturesTraining);
	Mat features, labels;
	std::vector<int> vecLabels;
    createTrainingData(features,labels,pathToPicturesTraining,vecLabels);
    cout << "created training data" << "Features: " << features.size() << std::endl;
    for(int labe: vecLabels){
    	cout << "Label: " << labe << std::endl; 
    }
    //Creat and train random forest
	RandomForest randForest;
	randForest.creat(100,1,4,10,2);
	cout << "created forest" << std::endl;
    randForest.train(features,labels,vecLabels);
    cout << "trained forest" << std::endl;
    //Load and creat bounding boxes on picture
    Mat picture;
    picture = imread("task3Data/test/0000.jpg",CV_LOAD_IMAGE_COLOR);
    if(!picture.data){
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    std::vector<vector<int> > boxes; 
    creatBoxesUsingSlidingWindow(PICTURE_ROW,PICTURE_COL,50,50,picture,boxes);
    cout << "Managed to creatBoxesUsingSlidingWindow" << std::endl;
    //Predict each bounding box
    HOGDescriptor hog(Size(PICTURE_COL,PICTURE_ROW), Size(BLOCK_COL,BLOCK_ROW), Size(BLOCK_STRID_COL,BLOCK_STRID_ROW), Size(CELL_COL,CELL_ROW),9);
    std::vector<pair<vector<int>, pair<int,int> > > boxPredictionPairs;
    randForest.evaluateBoxes(picture,hog,boxes,vecLabels,boxPredictionPairs);
    int i = 0;
    namedWindow("windo original picture", CV_WINDOW_AUTOSIZE);
    imshow("windo original picture",picture);
    //Draw bounding boxes, 01 is RED, 02 is GREE and 00 is BlUE
    for(pair<vector<int>, pair<int,int> > boxPredictionPair: boxPredictionPairs){
    	if(boxPredictionPair.second.first == 1 && boxPredictionPair.second.second > 0){
    		cout << "Prediction: " <<  boxPredictionPair.second.first << " prosent: " << boxPredictionPair.second.second << std::endl;
    		rectangle(picture,Point(boxPredictionPair.first[0],boxPredictionPair.first[2]),Point(boxPredictionPair.first[1],boxPredictionPair.first[3]),Scalar(0,0,255));
 		} else if (boxPredictionPair.second.first == 2 && boxPredictionPair.second.second > 0){
 			cout << "Prediction: " <<  boxPredictionPair.second.first << " prosent: " << boxPredictionPair.second.second << std::endl;
 			rectangle(picture,Point(boxPredictionPair.first[0],boxPredictionPair.first[2]),Point(boxPredictionPair.first[1],boxPredictionPair.first[3]),Scalar(0,255,0));
 		} else if (boxPredictionPair.second.first == 3 && boxPredictionPair.second.second > 0){
 			cout << "Prediction: " <<  boxPredictionPair.second.first << " prosent: " << boxPredictionPair.second.second << std::endl;
 			rectangle(picture,Point(boxPredictionPair.first[0],boxPredictionPair.first[2]),Point(boxPredictionPair.first[1],boxPredictionPair.first[3]),Scalar(255,0,0));	
		}

    }
    namedWindow("windo whit drawn boxes",CV_WINDOW_AUTOSIZE);
    imshow("windo whit drawn boxes",picture);
    cout << "foor loop done" << std::endl;  
    waitKey(0);
	return 0;
}
