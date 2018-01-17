#include <stdio.h>
#include <cstdlib>
#include "iostream"
#include <vector>
#include <string.h>
#include <fstream>
#include <dirent.h>
#include <string.h>

using namespace std;

void listFile();


int main(){
    listFile();
    return 0;
}
void listFile(){
        DIR *pDIR;
        struct dirent *entry;
        std::vector<string> dirs;
        std::vector<string> picturs;
        dirs.push_back("");
        string baseDir = "task2Data/train/";
        //for (string& addon: dirs){
        cout << "Size before loop: " << dirs.size() << std::endl;
        for(int i = 0; i <= dirs.size(); i++){
            cout << "Size in loop: " << dirs.size() << std::endl;
            const char* directory = baseDir.append(dirs[i]).c_str();
            if( pDIR=opendir(directory)){
                    while(entry = readdir(pDIR)){
                            //cout << "Hello World!" << std::endl;
                            if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 ){
                                //std::cout << entry->d_name << std::endl;
                                //picturs.push_back(entry->d_name);    
                                if (((string)entry->d_name).string::find_first_of(".")!=string::npos){
                                    //cout << "was a file" << std::endl;
                                    picturs.push_back(baseDir+dirs[i]+entry->d_name);
                                }
                                else{
                                    //cout << "was a folder" << std::endl;
                                    dirs.push_back(dirs[i]+entry->d_name+"/");
                                }
                            }

                    }
                    closedir(pDIR);
            }
        }
        //cout << "Second dir: " << dirs[1] << std::endl;
        for(string& picture: picturs){
            cout << "Picture: " << picture << std::endl;
        }
        for(string& dir: dirs){
            cout << "Dir: " << dir << std::endl;
        }
}