# MVS

## Dependencies
* OpenCV >= 2.4  
* CUDA >= 6.0   
## Usage
### CMakeLists.txt
Please modify line 20 of the CMakeLists.txt based on the GPU architecture of your device. Modify `arch=compute_80,code=sm_80` to `arch=compute_XX,code=sm_XX` and save. If you don't know the corresponding configuration of your device, please see [it](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).
```
20 set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_80,code=sm_80)
```
### Build
```
mkdir build  
cd build  
cmake ../src  
make  
```
### RUN

```
Use script colmap2mvsnet_acm.py to convert COLMAP SfM result to ACMMP input 
```
## Acknowledgemets
This code is developed on the basis of [ACMMP](https://github.com/GhiXu/ACMMP#acmmp) and [OpenMVS](https://github.com/cdcseacave/openMVS). Thanks to their authors for opening source of their excellent works.
