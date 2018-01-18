#include <stdio.h>
#include <cstdlib>
#include "iostream"
#include <vector>
#include <string.h>
#include <fstream>
#include <dirent.h>
#include <string.h>

using namespace std;

void reqursivelyFindJPG(string baseDir, std::vector<string>& picturs);


int main(){
    std::vector<string> picturs;
    string baseDir = "task2Data/train/";
    reqursivelyFindJPG(baseDir, picturs);
    for(string& picture: picturs){
       cout << "Picture: " << picture << std::endl;
    }
    return 0;
}
//A function that reqursively shershes a directory for a file with .jpg in tha name. 
//Every file it finds it stores in a vector the path to the file include the baseDir path
void reqursivelyFindJPG(string baseDir, std::vector<string>& picturs){
        DIR *pDIR;
        struct dirent *entry;
        std::vector<string> dirs;
        dirs.push_back("");
        cout << "Size before loop: " << dirs.size() << std::endl;
        for(int i = 0; i < dirs.size(); i++){
            string strDirectory = baseDir+dirs[i];
            const char* directory = strDirectory.c_str();
            if( pDIR=opendir(directory)){
                    while(entry = readdir(pDIR)){
                            cout << "Entry: " << entry->d_name << std::endl;
                            if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 ){   
                                if (((string)entry->d_name).string::find(".jpg")!=string::npos){
                                    picturs.push_back(baseDir+dirs[i]+entry->d_name);
                                }
                                else if (((string)entry->d_name).string::find_first_of(".")==string::npos){
                                    dirs.push_back(dirs[i]+entry->d_name+"/");
                                }
                            }
                    }
                    closedir(pDIR);
            }
        } 
        //for(string& dir: dirs){
        //    cout << "Dir: " << dir << std::endl;
        //}
}