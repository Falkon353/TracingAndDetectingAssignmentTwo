#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"
#include <string>
//#include "../hog_visualization.cpp"
#include "consts.hpp"
#include "utilities.hpp"
#include "iostream"
#include "randomForest.hpp"


/*int main(){
	Mat picture;
    picture = imread("task3Data/test/0000.jpg",CV_LOAD_IMAGE_COLOR);
    if(!picture.data){
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    resize(picture, picture, Size(128,96));
    namedWindow("Display windo picture", CV_WINDOW_AUTOSIZE);
	imshow("Display windo picture", picture);
	std::vector<vector<int> > boxes;
	creatBoxesUsingSlidingWindow(20,20,8,6,picture,boxes);
	cout << "managed to creatBoxesUsingSlidingWindow" << std::endl;
	rectangle(picture,Point(boxes[0][0],boxes[0][2]),Point(boxes[0][1],boxes[0][3]),Scalar(0,0,255));
	namedWindow("Display drawn box", CV_WINDOW_AUTOSIZE);
	imshow("Display drawn box", picture);
	cout << "First cordinate: " << boxes[0][0] << "," << boxes[0][2] << std::endl;
	cout << "Second cordinate: " << boxes[0][1] << "," << boxes[0][3] << std::endl;
	/*namedWindow("Display windo first box", CV_WINDOW_AUTOSIZE);
	imshow("Display windo first box", boxes[33]);
	namedWindow("Display windo last box", CV_WINDOW_AUTOSIZE);
	imshow("Display windo last box", boxes[37]);*/
	/*waitKey(0);
	return 0;
}*/


int main(){
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
	RandomForest randForest;
	randForest.creat(512,1,4,10,2);
	cout << "created forest" << std::endl;
    randForest.train(features,labels,vecLabels);
    cout << "trained forest" << std::endl;
    Mat picture;
    picture = imread("task3Data/test/0000.jpg",CV_LOAD_IMAGE_COLOR);
    cout << "Picture size clutered picture: " << picture.size() << std::endl;
    if(!picture.data){
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    std::vector<vector<int> > boxes;
    HOGDescriptor hog(Size(PICTURE_COL,PICTURE_ROW), Size(BLOCK_COL,BLOCK_ROW), Size(BLOCK_STRID_COL,BLOCK_STRID_ROW), Size(CELL_COL,CELL_ROW),9); 
    /*creatBoxesUsingSlidingWindow(PICTURE_ROW,PICTURE_COL,50,50,picture,boxes);
    cout << "Managed to creatBoxesUsingSlidingWindow" << std::endl;
    cout << "nrBoxes: " << boxes.size() << std::endl;
    std::vector<pair<vector<int>, pair<int,int> > > boxPredictionPairs;
    randForest.evaluateBoxes(picture,hog,boxes,vecLabels,boxPredictionPairs);
    cout << "nrBoxPredictionPairs: " << boxPredictionPairs.size() << std::endl;
    int i = 0;
    namedWindow("windo original picture", CV_WINDOW_AUTOSIZE);
    imshow("windo original picture",picture);
    for(pair<vector<int>, pair<int,int> > boxPredictionPair: boxPredictionPairs){
    	//cout << "Prediction: " <<  boxPredictionPair.second.first << " prosent: " << boxPredictionPair.second.second << std::endl;
    	if(boxPredictionPair.second.first == 1 && boxPredictionPair.second.second > 0){// and boxPredictionPair.second.second > 75){
    		//i++;
    		cout << "Prediction: " <<  boxPredictionPair.second.first << " prosent: " << boxPredictionPair.second.second << std::endl;
    		//cout << "First corinate: " << boxPredictionPair.first[0] << "," << boxPredictionPair.first[2] << std::endl;
    		//cout << "First corinate: " << boxPredictionPair.first[1] << "," << boxPredictionPair.first[3] << std::endl;
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
    cout << "foor loop done" << std::endl; */ 
    std::vector<vector<pair<int,int> > > answers;
    string baseDirExperiment = "task3Data/train/";
    randForest.predictFromPath(hog,baseDirExperiment,vecLabels,answers);
    for(vector<pair<int,int> > anwserFolder: answers){
    	cout << "New folder" << std::endl;
    	for(pair<int,int> answer: anwserFolder){
    		cout << "Answer: " << answer.first << " prosent: " << answer.second << std::endl;
    	}
    }
    waitKey(0);
	return 0;
}

/*int main(){
	Mat picture;
    picture = imread("task3Data/test/0000.jpg",CV_LOAD_IMAGE_COLOR);
    cout << "Picture size clutered picture: " << picture.size() << std::endl;
    if(!picture.data){
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    std::vector<vector<int> > boxes;
	creatBoxesUsingSlidingWindow(PICTURE_ROW,PICTURE_COL,50,50,picture,boxes);
    cout << "Managed to creatBoxesUsingSlidingWindow" << std::endl;
    cout << "nrBoxes: " << boxes.size() << std::endl;
    for(int i = 0; i < boxes.size(); i++){
    	Mat pictureBox = picture(Range(boxes[i][0],boxes[i][1]+1),Range(boxes[i][2],boxes[i][3]+1));
    	string name = "task3Data/experiment/" + to_string(i) + ".jpg";
    	try{
            cout << "Writing to path: " << name << std::endl;
            imwrite(name, pictureBox);
        }
        catch (runtime_error& ex){
          cout << "Was not able to store image" << std::endl;
          return 1;
        }
    }
	return 0;
}*/

/*int main()
{
	agmentPicturesInPath("task3Data/train/");
	return 0;
}*/