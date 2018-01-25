#ifndef UTILITIES_H
#define UTILITIES_H
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
#include "randomForest.hpp"
#include "consts.hpp"


using namespace std;
using namespace cv;

void reqursivelyFindJPG(string baseDir, std::vector< vector <string> >& pathToPictures);
int createTrainingData(Mat& features, Mat& labels, std::vector< vector<string> > pathToPictures, std::vector<int>& vecLabels);
int dTreePredict(HOGDescriptor& hog, ml::DTrees* dTree, std::vector<vector <string> > pathToPictures, std::vector<vector <int> >& answers);
int creatPicturDecriptor(Mat& mDescriptorPicture, HOGDescriptor& hog, string picturePath);
void creatBoxesUsingSlidingWindow(int nrRows, int nrColums, int shiftRows, int shiftColums, Mat& picture, std::vector<vector<int> >& boxes);
int agmentPicturesInPath(string baseDir);
#endif