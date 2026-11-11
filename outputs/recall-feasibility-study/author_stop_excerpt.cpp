#include <algorithm>
#include <cstddef>
float compute_regularization(int ox_y, float lambda, int kreg) {
    return lambda * std::max(0, ox_y - kreg);
}

struct Calibration {float lamhat; float regLambda; int kreg;};
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif
extern "C" API int author_stop(float score_k,float MAX_DISTANCE,float theta,float reg,int m,int rank){
 Calibration cal_params{theta,reg,1};int nlist=m;int ik=rank-1;
float max_reg_val = (1 + cal_params.regLambda * (nlist - cal_params.kreg)) + 10;
                            float nonconf_score = std::min(score_k / MAX_DISTANCE, 1.0f);
                            float reg_score_k = (1 - nonconf_score) + compute_regularization(ik + 1, cal_params.regLambda, cal_params.kreg);
                            reg_score_k = reg_score_k / max_reg_val;
                            
 return reg_score_k > cal_params.lamhat;
}
