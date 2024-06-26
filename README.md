# MVS
This code improves on [ACMMP](https://github.com/GhiXu/ACMMP#acmmp), and the main differences can be found in [MP-MVS:Multi-Scale Windows PatchMatch and Planar Prior Multi-View Stereo](https://arxiv.org/abs/2309.13294).

## Dependencies
* OpenCV >= 2.4  
* CUDA >= 6.0
* ncnn(option) If you want to generate sky masks
## Usage
### CMakeLists.txt
Please modify CMakeLists.txt based on the GPU architecture of your device. Modify `arch=compute_80,code=sm_80` to `arch=compute_XX,code=sm_XX` and save. If you don't know the corresponding configuration of your device, please see [it](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).
```
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_80,code=sm_80)
```
If ncnn is compiled, set the path：
```
set(ncnn_DIR "../ncnn/build/install/lib/cmake/ncnn")
```
### Build
```
git clone https://github.com/RongxuanTan/MP-MVS.git
cd MP-MVS
mkdir build  
cd build  
cmake ..    
make  
```

If you have ncnn installed, run the following command：
```
git clone https://github.com/RongxuanTan/MP-MVS.git
cd MP-MVS
mkdir build  
cd build  
cmake .. -D USE_NCNN=ON
make  
```
### Config.yaml
If it is an indoor scene (especially if it is a weakly textured scene), set it to 1. If it is an outdoor scene (including sky area), set it to 0. It is not recommended to use it unless sky area detection is added.  
`Geometric consistency planer prior`: Whether to use geometric consistency to construct planar prior model

The code and models for sky detection were derived from [Owner avatar Sky-Segmentation-and-Post-processing ](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing).
`Sky segment`: Whether to use sky area detection to eliminate sky artifacts

### Sky Filter
Before sky filter:
![Alt Text](result/1a.png)
![Alt Text](result/2a.png)
After sky filter:
![Alt Text](result/1b.png)
![Alt Text](result/2b.png)
### Run
```
//Use script colmap2mvsnet_acm.py to convert COLMAP SfM result to MPMVS input
python colmap2mvsnet_acm.py --dense_folder xxx --save_folder xxx
//Once in the build folder, execute the executable file
./MPMVS

```
## Acknowledgemets
This code is developed on the basis of [ACMMP](https://github.com/GhiXu/ACMMP#acmmp) and [OpenMVS](https://github.com/cdcseacave/openMVS). Thanks to their authors for opening source of their excellent works.
