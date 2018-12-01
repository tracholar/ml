#include "io.h"
#include "lr.cc"
#include<string.h>

using namespace std;

int main(int argc,char **argv){
    string fname;
    for(int i=0; i<argc; i++){
        if(strcmp(argv[i], "-f") == 0){
            fname = argv[i+1];
        }
    }
    ifstream fin;
    fin.open(fname, ios::in);
    if(!fin.is_open()){
        fprintf(stderr, "打开文件%s失败!\n", fname.c_str());
        exit(1);
    }
    ml::Data data;

    printf("读取数据...\n");
    ml::read_libsvm(fin, &data);
    printf("输出数据...\n");
    ml::to_libsvm(fopen("demo.out.txt", "w"), &data);

    ml::BinaryLogisticRegression lr;
    
    printf("训练模型...\n");
    lr.train(& data);

    printf("b = %.5g\n", lr.b);
    for(int i=0; i<lr.w.size(); i++) printf("%d: %.5g\n", i, lr.w[i]);

}