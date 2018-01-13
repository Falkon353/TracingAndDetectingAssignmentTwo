#include <stdio.h>
#include <cstdlib>
#include "iostream"
#include <vector>
#include <string.h>
#include <fstream>
#include <dirent.h>

using namespace std;

void listFile();


int main(){
    listFile();
    return 0;
}
void listFile(){
        DIR *pDIR;
        struct dirent *entry;
        std::vector<char*> dirs;
        if( pDIR=opendir("task2Data/train/00") ){
                while(entry = readdir(pDIR)){
                        if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 ){
                            std::cout << entry->d_name << std::endl;
                            dirs.push_back(entry->d_name);    
                        }
                        

                }
                closedir(pDIR);
        }
        //cout << dirs << std::endl;
        cout << dirs[0] << std::endl;
}