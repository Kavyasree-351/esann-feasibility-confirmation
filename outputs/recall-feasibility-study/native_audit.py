"""Compile actual author scalar early-stop expressions, NOT the full engine."""
import argparse,ctypes,hashlib,json,re,subprocess
from pathlib import Path
import numpy as np
from prepare import ROOT,WORK,save,digest
from engine import score_value
def main():
    source=WORK/'conann-upstream/conann/faiss/IndexIVF.cpp';text=source.read_text()
    signature='float IndexIVF::compute_regularization(int ox_y, float lambda, int kreg) const'
    start=text.index(signature);end=text.index('\n}',start)+2
    regularization=text[start:end].replace(signature,'float compute_regularization(int ox_y, float lambda, int kreg)')
    start=text.index('float max_reg_val = (1 + cal_params.regLambda',text.index('// NOTE: early stopping mode'))
    end=text.index('if (reg_score_k > cal_params.lamhat)',start)
    expressions=text[start:end]
    work=WORK/'conann-native-audit';work.mkdir(exist_ok=True)
    cpp=work/'author_stop.cpp'
    cpp.write_text('#include <algorithm>\n#include <cstddef>\n'+regularization+'''\n
struct Calibration {float lamhat; float regLambda; int kreg;};
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif
extern "C" API int author_stop(float score_k,float MAX_DISTANCE,float theta,float reg,int m,int rank){
 Calibration cal_params{theta,reg,1};int nlist=m;int ik=rank-1;
'''+expressions+'\n return reg_score_k > cal_params.lamhat;\n}\n')
    exe=work/'author_stop.dll'
    cmd=['g++','-shared','-O2','-static-libgcc','-static-libstdc++',str(cpp),'-o',str(exe)]
    subprocess.run(cmd,check=True,capture_output=True)
    lib=ctypes.CDLL(str(exe));fn=lib.author_stop;fn.restype=ctypes.c_int
    fn.argtypes=[ctypes.c_float]*4+[ctypes.c_int]*2
    rng=np.random.default_rng(813);n=10000
    for _ in range(n):
        kth=float(rng.uniform(0,6));reg=float(rng.choice([0,.001,.01,.1]));j=int(rng.integers(1,65));theta=float(rng.uniform(0,1))
        assert bool(fn(kth,4,theta,reg,64,j))==bool(score_value(kth,reg,j)>np.float32(theta))
    # Mechanism control: returning p=1 requires visiting prefix2 before rollback.
    kth=np.array([3.,2.,1.]);sc=score_value(kth,0.,np.arange(1,4),3);theta=float((sc[0]+sc[1])/2)
    assert not fn(kth[0],4,theta,0,3,1) and fn(kth[1],4,theta,0,3,2)
    commit=subprocess.check_output(['git','-C',str(WORK/'conann-upstream'),'rev-parse','HEAD'],text=True).strip()
    save(ROOT/'NATIVE_AUDIT.json',dict(author_commit=commit,source_sha256=digest(source),
        extracted_cpp_sha256=digest(cpp),random_differential_cases=n,lookahead_positive_control=True,
        full_author_engine_built=False,scope='Actual extracted scalar expressions compiled; list scanner/calibration/native Faiss engine not reproduced.',
        build_blocker='Full engine CMake configuration: required BLAS/MKL library not found.',
        upstream_url='https://github.com/soniahorchidan/conann',full_engine_claim_allowed=False))
    (ROOT/'author_stop_excerpt.cpp').write_bytes(cpp.read_bytes())
    (ROOT/'CONANN_LICENSE.txt').write_bytes((WORK/'conann-upstream/LICENSE').read_bytes())
    (ROOT/'FAISS_LICENSE.txt').write_bytes((WORK/'conann-upstream/conann/LICENSE').read_bytes())
    print('Author scalar kernel: 10000 differential cases and lookahead control passed. Full engine remains unbuilt.')
if __name__=='__main__':main()
