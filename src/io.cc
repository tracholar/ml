#include "io.h"


namespace ml {

    /**
     *  y = NULL 表示没有label 
     */
    inline void parse_libsvm(const char *s, float *y, SparseVector * v){
        if(s == NULL || v == NULL) return;
        int nchar;
        if(y != NULL){
            if(sscanf(s, "%f%n", y, &nchar) >= 1){
                s += nchar;
            }else{
                fprintf(stderr, "输入格式有误：%s", s);
                exit(1);
            }
        }
        int idx; float val;
        v->idx.clear();
        v->val.clear();
        while(sscanf(s, "%d:%f%n", &idx, &val, &nchar) >= 2){
            s += nchar;
            assert(idx >= 0);
            v->idx.push_back(idx);
            v->val.push_back(val);
        }
    }

    void to_libsvm(FILE * fp, Data * dptr){
        for(int i=0; i<dptr->x.size(); i++){
            if(i < dptr->y.size()){
                fprintf(fp, "%g ", dptr->y[i]);
            }
            for(int j=0; j<dptr->x[i].idx.size(); j++){
                if(j > 0) fputc(' ', fp);
                fprintf(fp, "%d:%.5g", dptr->x[i].idx[j], dptr->x[i].val[j]);
            }
            fputc('\n', fp);
        }
    }

    void read_libsvm(std::ifstream & fin, Data * dptr, bool label, int batch_size){
        dptr->y.clear();
        dptr->x.clear();
        
        float y;
        SparseVector x;
        std::string line;
        assert(dptr != NULL);
            
        while(!fin.eof()){
            if(batch_size > 0 && dptr->x.size() >= batch_size) break;
            
            std::getline(fin, line);
            if(line.length() == 0) continue;
            const char * pline = line.c_str();
            
            if(label){
                parse_libsvm(pline, &y, &x);
                dptr->y.push_back(y);
            }else{
                parse_libsvm(pline, NULL, &x);
            }
            
            dptr->x.push_back(x);
            
        }
    }
}