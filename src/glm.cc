#include "io.h"
#include "lr.cc"
#include<string.h>

using namespace std;

int main(int argc,char **argv){
    string fname, output = "stdout";
    ml::GLMCONF c;

    for(int i=0; i<argc; i++){
        if(strcmp(argv[i], "-f") == 0){
            fname = argv[i+1];
            i+= 1;
        }
        if(strcmp(argv[i], "-o") == 0){
            output = argv[i+1];
            i+= 1;
        }
        if(strcmp(argv[i], "-iter") == 0){
            c.max_iter = atoi(argv[i+1]);
            i+= 1;
        }
        if(strcmp(argv[i], "-reg1") == 0){
            c.reg1 = atof(argv[i+1]);
            i+= 1;
        }
        if(strcmp(argv[i], "-reg2") == 0){
            c.reg2 = atof(argv[i+1]);
            i+= 1;
        }
        if(strcmp(argv[i], "-lr") == 0){
            c.lr = atof(argv[i+1]);
            i+= 1;
        }
        if(strcmp(argv[i], "-obj") == 0){
            c.obj = argv[i+1];
            i+= 1;
        }
        if(strcmp(argv[i], "-v") == 0){
            c.verbose = true;
        }
        if(strcmp(argv[i], "-h") == 0){
            printf("用法： glm [-h] [-f data.txt] [-iter 100] [-reg1 0.001] [-reg2 0.001] [-lr 1] [-v] [-o output] [-obj logloss]\n");
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
    fprintf(stderr, "读取数据...\n");
    ml::read_libsvm(fin, &data);

    time_t t2 = time(NULL);
    fprintf(stderr, "读取数据耗时 %ld s\n", t2 - t1);

    //printf("输出数据...\n");
    //ml::to_libsvm(fopen("demo.out.txt", "w"), &data);

    ml::GLM lr;
    
    fprintf(stderr, "训练模型...\n");
    lr.train(& data, c);

    time_t t3 = time(NULL);
    fprintf(stderr, "训练模型耗时 %ld s\n", t3 - t2);

    FILE * fp;
    if(output == "stdout"){
        fp = stdout;
    }else{
        fp = fopen(output.c_str(), "w");
        if(fp == NULL){
            fprintf(stderr, "打开文件 %s 失败!\n", output.c_str());
            exit(1);
        }
    }
    fprintf(fp, "b: %.5g\n", lr.b);
    for(int i=0; i<lr.w.size(); i++) fprintf(fp, "%d: %.5g\n", i, lr.w[i]);

}