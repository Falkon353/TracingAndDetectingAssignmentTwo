#include <stdio.h>
#include <cstdlib>
#include "iostream"
#include <vector>
#include <string.h>
#include <fstream>
#include <dirent.h>
#include <string.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;

void reqursivelyFindJPG(string baseDir, std::vector< vector <string> >& pathToPictures);
int createTrainingData(Mat& features, Mat& labels, std::vector< vector<string> > pathToPictures);
int dTreePredict(HOGDescriptor& hog, ml::DTrees& dTree, std::vector<vector <string> > pathToPictures, std::vector<vector <int> >& answer);


/*int main(){
   std::vector< vector<string> > pathToPictures;
   string baseDir = "task2Data/train/";
   reqursivelyFindJPG(baseDir, pathToPictures);
   for(vector<string>& paths: pathToPictures){
      cout << "Hello World!" << std::endl;
      for(string& picturPath: paths){
          cout << "Picture: " << picturPath << std::endl;
      }
   }
   /*Mat features, labels;
   createTrainingData(features,labels,pathToPictures);
   cout << "features size" << features.size() << std::endl;
   cout << "labels size" << labels.size() << std::endl;*/
   /*return 0;
}*/
//A function that reqursively shershes a directory for a file with .jpg in tha name. 
//Every file it finds it stores in a vector the path to the file include the baseDir path
void reqursivelyFindJPG(string baseDir, std::vector< vector <string> >& pathToPictures){
        DIR *pDIR;
        struct dirent *entry;
        std::vector<string> dirs;
        dirs.push_back("");
        for(int i = 0; i < dirs.size(); i++){
            std::vector<string> pictursInDir;
            string strDirectory = baseDir+dirs[i];
            const char* directory = strDirectory.c_str();
            if( pDIR=opendir(directory)){
                while(entry = readdir(pDIR)){
                        if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 ){   
                            if (((string)entry->d_name).string::find(".jpg")!=string::npos){
                                pictursInDir.push_back(baseDir+dirs[i]+entry->d_name);
                                //cout << "Picture Path: " << baseDir+dirs[i]+entry->d_name << std::endl;
                            }
                            else if (((string)entry->d_name).string::find_first_of(".")==string::npos){
                                dirs.push_back(dirs[i]+entry->d_name+"/");
                            }
                        }
                }
                closedir(pDIR);
            }
            if (pictursInDir.size() != 0){
                pathToPictures.push_back(pictursInDir);
            }
        } 
        for(string& dir: dirs){
           cout << "Dir: " << dir << std::endl;
        }
}

int createTrainingData(Mat& features, Mat& labels, std::vector< vector<string> > pathToPictures){
    int classCounter = 1;
    HOGDescriptor hog(Size(128,96), Size(16,12), Size(8,6), Size(16,12),9);
    for(vector<string>& paths: pathToPictures){
        for(string& picturPath: paths){
            //cout << "Picture: " << picturPath << std::endl;
            Mat picture;
            picture = imread(picturPath,CV_LOAD_IMAGE_COLOR);
            if(!picture.data){
                cout << "Could not open or find the image" << std::endl;
                return -1;
            }
            resize(picture, picture, Size(128,96));
            std::vector<float> descriptorPicture;
            hog.compute(picture,descriptorPicture);
            Mat mDescriptorPicture = Mat(1, descriptorPicture.size(), CV_32FC1);
            memcpy(mDescriptorPicture.data, descriptorPicture.data(), descriptorPicture.size());
            features.push_back(mDescriptorPicture);
            labels.push_back(classCounter);
        }
        classCounter++;
    }
    return 0;
}

int dTreePredict(HOGDescriptor& hog, ml::DTrees* dTree, std::vector<vector <string> > pathToPictures, std::vector<vector <int> >& answers){
    for(vector<string>& paths: pathToPictures){
        std::vector<int> folderAnswers;
        for(string& picturePath: paths){ 
            Mat picture;
            picture = imread(picturePath,CV_LOAD_IMAGE_COLOR);
            if(!picture.data){
                cout << "Could not open or find the image" << std::endl;
                return -1;
            }
            resize(picture, picture, Size(128,96));
            std::vector<float> descriptorPicture;
            hog.compute(picture,descriptorPicture);
            Mat mDescriptorPicture = Mat(1, descriptorPicture.size(), CV_32FC1);
            memcpy(mDescriptorPicture.data, descriptorPicture.data(), descriptorPicture.size());
            int answer = dTree -> predict(mDescriptorPicture);
            folderAnswers.push_back(answer);
        }
        answers.push_back(folderAnswers);
    }
    return 0;

}