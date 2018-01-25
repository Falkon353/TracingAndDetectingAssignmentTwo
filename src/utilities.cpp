#include "utilities.hpp"


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
    HOGDescriptor hog(Size(PICTURE_COL,PICTURE_ROW), Size(BLOCK_COL,BLOCK_ROW), Size(BLOCK_STRID_COL,BLOCK_STRID_ROW), Size(CELL_COL,CELL_ROW),9);
    int i = 0;
    for(vector<string>& paths: pathToPictures){
        for(string& picturePath: paths){
            i++;
            Mat mDescriptorPicture;
            creatPicturDecriptor(mDescriptorPicture,hog,picturePath);
            features.push_back(mDescriptorPicture);
            labels.push_back(classCounter);
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
            Mat mDescriptorPicture;
            creatPicturDecriptor(mDescriptorPicture,hog,picturePath);
            int answer = dTree -> predict(mDescriptorPicture);
            folderAnswers.push_back(answer);
        }
        answers.push_back(folderAnswers);
    }
    return 0;
}

int creatPicturDecriptor(Mat& mDescriptorPicture, HOGDescriptor& hog, string picturePath){
    Mat picture;
    picture = imread(picturePath,CV_LOAD_IMAGE_COLOR);
    if(!picture.data){
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    resize(picture, picture, Size(PICTURE_COL,PICTURE_ROW));
    std::vector<float> descriptorPicture;
    hog.compute(picture,descriptorPicture);
    mDescriptorPicture = Mat(1, descriptorPicture.size(), CV_32FC1);
    memcpy(mDescriptorPicture.data, descriptorPicture.data(), descriptorPicture.size());
    return 0;
}

void creatBoxesUsingSlidingWindow(int nrRows, int nrColums, int shiftRows, int shiftColums, Mat& picture, std::vector<vector<int> >& boxes){
    for(int botomRow = nrRows; botomRow < picture.rows; botomRow += shiftRows){
        for(int rightColum = nrColums; rightColum < picture.cols; rightColum += shiftColums){
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