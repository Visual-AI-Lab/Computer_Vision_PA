import matlab.engine
import numpy as np
import time
import cv2

# https://kr.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
# Window/Ubuntu terminal에서 "{Matlab 설치 경로}/extern/engines/python"으로 이동 후,
# python setup.py install 실행

# https://mscipio.github.io/post/matlab-from-python/

def main():

    eng = matlab.engine.start_matlab()
    eng.addpath(r'/home/dongmin/2022_GIST_lec_CV-PA/PA2/Step2/', nargout=0) # 'calibrated_fivepoint.m'가 위치한 경로
    end = time.time()
    for i in range(100):
        a = np.random.rand(3,5).tolist()
        a = matlab.double(a)
        b = np.random.rand(3,5).tolist()
        b = matlab.double(b)
        E = eng.calibrated_fivepoint(a, b)
        print(np.asarray(E))
    print(time.time()-end)

    eng.quit()


if __name__ == '__main__':
    main()

