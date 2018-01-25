#include "utilities.hpp"


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

int createTrainingData(Mat& features, Mat& labels, std::vector< vector<string> > pathToPictures, std::vector<int>& vecLabels){
    int classCounter = 1;
    //HOGDescriptor hog(Size(128,96), Size(16,12), Size(8,6), Size(16,12),9);
    HOGDescriptor hog(Size(HOG_WINDOW_COL,HOG_WINDOW_ROW), Size(BLOCK_COL,BLOCK_ROW), Size(BLOCK_STRID_COL,BLOCK_STRID_ROW), Size(CELL_COL,CELL_ROW),9);
    int i = 0;
    for(vector<string>& paths: pathToPictures){
        for(string& picturePath: paths){
            i++;
            vector<Mat> vectorDescriptorsPicture;
            creatPicturDecriptor(vectorDescriptorsPicture,hog,picturePath);
            for(Mat mDescriptorPicture: vectorDescriptorsPicture){
                features.push_back(mDescriptorPicture);
                labels.push_back(classCounter);
            }
        }

        vecLabels.push_back({classCounter});
        classCounter++;
    }
    cout << "Nr picturs feuures extracted: " << i << std::endl;
    return 0;
}

int dTreePredict(HOGDescriptor& hog, ml::DTrees* dTree, std::vector<vector <string> > pathToPictures, std::vector<vector <int> >& answers){
    for(vector<string>& paths: pathToPictures){
        std::vector<int> folderAnswers;
        for(string& picturePath: paths){ 
            vector<Mat> vectorDescriptorsPicture;
            creatPicturDecriptor(vectorDescriptorsPicture,hog,picturePath);
            int answer;
            for(Mat mDescriptorPicture: vectorDescriptorsPicture){
                answer = dTree -> predict(mDescriptorPicture);
                folderAnswers.push_back(answer);
            }
        }
        answers.push_back(folderAnswers);
    }
    return 0;
}

int creatPicturDecriptor(vector<Mat>& vectorDescriptorsPicture, HOGDescriptor& hog, string picturePath){
    Mat picture;
    picture = imread(picturePath,CV_LOAD_IMAGE_COLOR);
    if(!picture.data){
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    //resize(picture, picture, Size(PICTURE_COL,PICTURE_ROW));
    for(int j = 0; j < 3; j++){
        Mat patch;
        if(j == 0){
            patch = picture(Range(picture.rows/2,picture.rows/2+HOG_WINDOW_ROW),Range(10,10+HOG_WINDOW_COL));
        } else if (j == 1){
            patch = picture(Range(10,10+HOG_WINDOW_ROW),Range(picture.cols/2,picture.cols/2+HOG_WINDOW_COL));
        } else{
            patch = picture(Range(picture.rows-(10+HOG_WINDOW_ROW),picture.rows-10),Range(picture.cols/2,picture.cols/2+HOG_WINDOW_COL));
        }
        //cout << "Size patch: " << patch.size() << std::endl;
        std::vector<float> descriptorPicture;
        hog.compute(patch,descriptorPicture);
        Mat mDescriptorPicture = Mat(1, descriptorPicture.size(), CV_32FC1);
        memcpy(mDescriptorPicture.data, descriptorPicture.data(), descriptorPicture.size());
        vectorDescriptorsPicture.push_back(mDescriptorPicture);
    }
    return 0;
}

void creatBoxesUsingSlidingWindow(int nrRows, int nrColums, int shiftRows, int shiftColums, Mat& picture, std::vector<vector<int> >& boxes){
    for(int botomRow = nrRows; botomRow < picture.rows; botomRow += shiftRows){
        for(int rightColum = nrColums; rightColum < picture.cols; rightColum += shiftColums){
            //cout << "botomRow: " << botomRow << " rightColum: " << rightColum << std::endl;
            std::vector<int> box = {botomRow-nrRows, botomRow, rightColum-nrColums, rightColum};
            boxes.push_back(box);
        }
    }
}


int agmentPicturesInPath(string baseDir){
    std::vector<vector<string> > pathToPictures;
    reqursivelyFindJPG(baseDir,pathToPictures);
    int i = 0;
    for(std::vector<string> paths: pathToPictures){
        for(string picturePath: paths){
            cout << "picturePath: " << picturePath << std::endl;
            Mat picture;
            picture = imread(picturePath,CV_LOAD_IMAGE_COLOR);
            //cout << "picture size: " << picture.size() << std::endl;
            if(!picture.data){
                cout << "Could not open or find the image" << std::endl;
                return -1;
            }
            Mat augmentedPicture;
            if(i%3 == 0){
                rotate(picture,augmentedPicture,ROTATE_90_CLOCKWISE);
            }else if(i%3 == 1){
                rotate(picture,augmentedPicture,ROTATE_90_COUNTERCLOCKWISE);
            } else{
                flip(picture,augmentedPicture,1);
            }
            string newName = picturePath.substr(0,picturePath.length()-4)+"a.jpg";
            try{
                cout << "Writing to path: " << newName << std::endl;
                imwrite(newName, augmentedPicture);
            }
            catch (runtime_error& ex){
              cout << "Was not able to store image" << std::endl;
              return 1;
            }
        }
    }
    return 0;
}