#include "io.h"
#include<math.h>

namespace ml{
    class BinaryLogisticRegression {        
    public:
        std::vector<float> w;
        float b;

        inline float sigmoid(float x){
            return 1.0/(1 + exp(-x));
        }
        void loss_function(Data * dptr, float * loss, std::vector<float> * dw, float * db){
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
                p = sigmoid(p);
                //fprintf(stderr, "%d %f\n", i, p);

                if(dw != NULL && db != NULL){
                    float dp = p - dptr->y[i];
                    if(dptr->y[i] == 1) *loss += - log(p);
                    else *loss += -log(1-p);
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

        void train(Data * dptr, float lr=1, float reg2=1e-2, float reg1=1e-3, int max_iter=100, bool verbose = true){
            // init
            init(dptr);
            std::vector<float> dw(w.size());
            float db = 0, loss, old_loss = 1e10;

            if(verbose){
                fprintf(stderr, "iter\tloss\tlr\n");
            }
            for(int i=0; i<max_iter; i++){
                loss_function(dptr, &loss, &dw, &db);
                b -= lr * (db + reg2 * b);
                for(int j=0; j<dw.size(); j++) {
                    w[j] -= lr * (dw[j] + reg2 * w[j]);
                    if(reg1 > 0 && w[j] > - reg1*lr && w[j] < reg1*lr) w[j] = 0;
                }

                if(verbose){
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