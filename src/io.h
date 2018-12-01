#ifndef ML_IO_H_
#define ML_IO_H_

#include<iostream>
#include<vector>
#include<string>
#include<exception>
#include <fstream>
#include<assert.h>

namespace ml {
    struct SparseVector {
        std::vector<int> idx;
        std::vector<float> val;
    };

    struct Data {
        std::vector<float> y;
        std::vector<SparseVector> x;
    };

    /**
     *  y = NULL 表示没有label 
     */
    inline void parse_libsvm(char *s, float *y, SparseVector * v);

    void read_libsvm(std::ifstream & fin, Data * dptr, bool label = true, int batch_size = -1);

    void to_libsvm(FILE * fp, Data * dptr);
}

#endif