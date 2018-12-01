#include "io.h"
#include "lr.cc"
#include<string.h>

using namespace std;

int main(int argc,char **argv){
    string fname;
    int iter = 100;
    float reg1 = 1e-3, reg2 = 1e-3, learning_rate = 1;
    bool verbose = false;
    for(int i=0; i<argc; i++){
        if(strcmp(argv[i], "-f") == 0){
            fname = argv[i+1];
            i+= 1;
        }
        if(strcmp(argv[i], "-iter") == 0){
            iter = atoi(argv[i+1]);
            i+= 1;
        }
        if(strcmp(argv[i], "-reg1") == 0){
            reg1 = atof(argv[i+1]);
            i+= 1;
        }
        if(strcmp(argv[i], "-reg2") == 0){
            reg2 = atof(argv[i+1]);
            i+= 1;
        }
        if(strcmp(argv[i], "-lr") == 0){
            learning_rate = atof(argv[i+1]);
            i+= 1;
        }
        if(strcmp(argv[i], "-v") == 0){
            verbose = true;
        }
        if(strcmp(argv[i], "-h") == 0){
            printf("用法： lr [-h] [-f data.txt] [-iter 100] [-reg1 0.001] [-reg2 0.001] [-lr 1] [-v]\n");
            exit(0);
        }
    }
    ifstream fin;
    fin.open(fname, ios::in);
    if(!fin.is_open()){
        fprintf(stderr, "打开文件%s失败!\n", fname.c_str());
        exit(1);
    }
    ml::Data data;

    time_t t1 = time(NULL);
    printf("读取数据...\n");
    ml::read_libsvm(fin, &data);

    time_t t2 = time(NULL);
    printf("读取数据耗时 %ld s\n", t2 - t1);

    //printf("输出数据...\n");
    //ml::to_libsvm(fopen("demo.out.txt", "w"), &data);

    ml::BinaryLogisticRegression lr;
    
    printf("训练模型...\n");
    lr.train(& data, learning_rate, reg2, reg1, iter, verbose);

    time_t t3 = time(NULL);
    printf("训练模型耗时 %ld s\n", t3 - t2);

    printf("b = %.5g\n", lr.b);
    for(int i=0; i<lr.w.size(); i++) printf("%d: %.5g\n", i, lr.w[i]);

}