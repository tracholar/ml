#include "io.h"
#include<math.h>

typedef void(* OBJECT_FUNCTION)(float, float, float*, float*);

namespace ml{
    inline float sigmoid(float x){
        return 1.0/(1 + exp(-x));
    }
    inline void object_function_logloss(float score, float y, float *logloss, float *gradient){
        float p = sigmoid(score);
        *gradient = p - y;
        if(y == 1) *logloss = - log(p);
        else *logloss = -log(1-p);
    }
    inline void object_function_mse(float score, float y, float *loss, float *gradient){
        *gradient = score - y;
        *loss = 0.5*(*gradient)*(*gradient);
    }
    inline void object_function_hinge(float score, float y, float *loss, float *gradient){
        if(y == 0) y =-1;
        *loss = 1 - y * score;
        if(*loss < 0){
            *loss = 0;
            *gradient = 0;
        } else{
            *gradient = - y;
        }
    }

    struct ProblemConf {
        float lr=1;
        float reg2=1e-2;
        float reg1=1e-3;
        int max_iter=100;
        bool verbose = true;
        std::string obj = "logloss";
    };
        
    class BinaryLogisticRegression {        
    public:
        std::vector<float> w;
        float b;

        void loss_function(Data * dptr, float * loss, std::vector<float> * dw, float * db, OBJECT_FUNCTION obj_func = object_function_logloss){
            assert(dptr->x.size() == dptr->y.size());

            *loss = 0;
            if(dw != NULL && db != NULL){
                for(int i=0; i<dw->size(); i++) (*dw)[i] = 0;
                *db = 0;
            }
            for(int i=0; i<dptr->x.size(); i++){
                float p = b;
                for(int j=0; j<dptr->x[i].idx.size(); j++){
                    int idx = dptr->x[i].idx[j];
                    float val = dptr->x[i].val[j];
                    p += w[idx] * val;
                }
                //p = sigmoid(p);
                //fprintf(stderr, "%d %f\n", i, p);

                if(dw != NULL && db != NULL){
                    float dp, lossi;
                    obj_func(p, dptr->y[i], &lossi, &dp);
                    *loss += lossi;
                    for(int j=0; j<dptr->x[i].idx.size(); j++){
                        int idx = dptr->x[i].idx[j];
                        float val = dptr->x[i].val[j];
                        (*dw)[idx] += dp * val;
                    }
                    *db += dp;
                }
            }
            for(int i=0; i<dw->size(); i++) (*dw)[i] /= dptr->x.size();
            *db /= dptr->x.size();
            *loss /= dptr->x.size();
        }

        void init(Data *dptr){
            int n_feature = 0;
            for(int i=0; i<dptr->x.size(); i++){
                if(dptr->x[i].idx.size() > 0 && n_feature <= dptr->x[i].idx[ dptr->x[i].idx.size()-1 ]){
                    n_feature = dptr->x[i].idx[ dptr->x[i].idx.size()-1 ] + 1;
                }
            }
            for(int i=0; i<n_feature; i++){
                w.push_back(0.0);
            }
            b = 0;
        }

        void train(Data * dptr, ProblemConf conf){
            // init
            init(dptr);
            std::vector<float> dw(w.size());
            float db = 0, loss, old_loss = 1e10;

            if(conf.verbose){
                fprintf(stderr, "iter\tloss\tlr\n");
            }
            float lr = conf.lr, reg1 = conf.reg1, reg2 = conf.reg2;
            OBJECT_FUNCTION cb;
            if(conf.obj == "logloss"){
                cb = object_function_logloss;
            }else if(conf.obj == "mse"){
                cb = object_function_mse;
            }else if(conf.obj == "hinge"){
                cb = object_function_hinge;
            }else{
                fprintf(stderr, "不支持的目标函数类型：%s", conf.obj.c_str());
                exit(1);
            }
            for(int i=0; i<conf.max_iter; i++){
                loss_function(dptr, &loss, &dw, &db, cb);
                b -= lr * (db + reg2 * b);
                for(int j=0; j<dw.size(); j++) {
                    w[j] -= lr * (dw[j] + reg2 * w[j]);
                    if(reg1 > 0 && w[j] > - reg1*lr && w[j] < reg1*lr) w[j] = 0;
                }

                if(conf.verbose){
                    fprintf(stderr, "%3d\t%.6f\t%.5g\n", i, loss, lr);
                }

                /*
                if(loss > old_loss && lr > 1e-3) lr /= 3;
                else if(abs(old_loss - loss)/loss < 1e-5 && lr < 100) lr *= 3;
                old_loss = loss;
                */
                
            }
        }
    };
}