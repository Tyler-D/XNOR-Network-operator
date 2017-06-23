#include<iostream>
#include<stdint.h>
#include<string>
using namespace std;
#define BIN_SIZE 64
int main(int argc, char** argv){
    string str = argv[1];
    uint64_t num = 0;
    uint64_t multi = 1;
    for(int i = str.size()-1; i >= 0; i--){
        num += multi*(str[i]-'0');
        multi *= 10;
    }
    cout <<num<<endl;
    int cnt = 0;
    while(cnt != BIN_SIZE){
      cout<<(num&1);
      num = num>>1;
      cnt ++;
    }
    cout<<endl;
}
