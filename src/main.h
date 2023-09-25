#ifndef _MAIN_H_
#define _MAIN_H_

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include "iomanip"
#include <math.h>

#include <sys/stat.h> // mkdir
#include <sys/types.h> // mkdir
#include <sys/time.h>
#include <unistd.h>

#include "PatchMatch.h"
#include "utility.h"

class Time
{
public:
    Time(){}
    ~Time(){}
 
    void start()
    {
        gettimeofday(&tv1,nullptr);
    }
 
    float cost()
    {
        gettimeofday(&tv2,nullptr);
        return (1000000*(tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec));
    }
private:
    struct timeval tv1, tv2;
};


#endif // _MAIN_H_