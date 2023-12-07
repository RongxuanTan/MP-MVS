#ifndef _PATCHMATCH_H_
#define _PATCHMATCH_H_
#include <deque>
#include <thread>
#include <mutex>
#include <cstdarg>

#include <vector_types.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"

#include "main.h"
#include "utility.h"

#ifdef BUILD_NCNN
#include "SkyRegionDetect.h"
#endif

struct PointList {
    float3 coord;
    float3 normal;
    float3 color;
};

struct Camera
{
    /* data */
    float K[9];
    float R[9];
    float t[3];
    float C[3];
    int height;
    int width;
    float depth_min;
    float depth_max;    
};

struct PatchMatchParams {
    int max_iterations = 3;
    int nSizeHalfWindow = 5;
    int num_images = 5;
    int max_image_size=3200;
    int nSizeStep = 2;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    int top_k = 4;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    int max_scale=2;

    float scaled_cols;
    float scaled_rows;

    bool geom_consistency = false;
    bool geomPlanarPrior=false;
    bool planar_prior = false;
};

struct Triangle {
    cv::Point pt1, pt2, pt3;
    Triangle (const cv::Point _pt1, const cv::Point _pt2, const cv::Point _pt3) : pt1(_pt1) , pt2(_pt2), pt3(_pt3) {}
};
void GenerateSkyRegionMask(std::vector<Scene> &Scenes, std::string &project_path, std::string &dense_folder, const int max_image_size);
void checkCudaCall(const cudaError_t error);
void GenerateSampleList(const ConfigParams &config, std::vector<Scene> &Scenes);
Camera ReadCamera(const std::string &cam_path);
void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc);
void ProcessProblem(const std::string &input_folder ,const std::string &output_folder, std::vector<Scene> &Scenes, const int ID,bool geom_consistency, bool planar_prior);
float3 Get3DPointonRefCam(const int x, const int y, const float depth, const Camera camera);
float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera);
cv::Vec3f TransformNormalonWorld(const Camera camera, cv::Vec3f normal);
float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2);
void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth);
void  RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera);
void RunFusion(const std::string &dense_folder, const std::string &out_folder, const std::vector<Scene> &Scenes, bool sky_mask, bool use_prior_map);

class PatchMatchCUDA
{
private:
    int num_img;
    std::vector<cv::Mat> images;
    std::vector<cv::Mat_<float>> depths;
    std::vector<Camera> cameras;    

    std::vector<cudaArray_t> cudaImageArrays;
    std::vector<cudaTextureObject_t> textureImages; 
    std::vector<cudaArray_t> cudaDepthArrays;   
    std::vector<cudaTextureObject_t> textureDepths;

    cudaTextureObject_t* cudaTextureImages;
    cudaTextureObject_t* cudaTextureDepths;
    float4 *cudaPlaneHypotheses;
    float4 *hostPlaneHypotheses;
    Camera *cudaCameras;
    float *cudaCosts;
    float *hostCosts;
    curandState* cudaRandStates;
    unsigned int *cudaSelectedViews;
    float4 *hostPriorPlanes;
    float4 *cudaPriorPlanes;
    unsigned int *hostPlaneMask; 
    unsigned int *cudaPlaneMask;

    uchar *cudaTexCofMap;
    uchar *hostTexCofMap;
    float *cudaGeomCosts;
    float *hostGeomCosts;

    PatchMatchParams params;

    std::string input_folder;
    std::string output_folder;
public:

    //.cpp
    float GetAngleDiff(const cv::Point tri, const float4 n4, const int width);
    void SetGeomConsistencyParams(bool geom_consistency,bool planar_prior);
    void SetPlanarPriorParams();
    void SetFolder(const std::string &_input_folder,const std::string &_output_folder);
    void PatchMatchInit(std::vector<Scene> Scenes,const int ID);
    void DataInit();
    void AllocatePatchMatch();
    void CudaMemInit(Scene &scene);
    void CudaPlanarPriorInitialization(const std::vector<float4> &PlaneParams, const cv::Mat_<float> &masks);

    float GetDepthFromPlaneParam(const float4 plane_hypothesis, const int x, const int y);
    float GetMinDepth();
    float GetMaxDepth();
    int GetReferenceImageWidth();
    int GetReferenceImageHeight();
    cv::Mat GetReferenceImage(); 
    float4 GetPlaneHypothesis(const int index);
    float GetCost(const int index);
    uchar GetTextureCofidence(const int index);
    float GetGeomCost(const int index);

    float4 GetPriorPlaneParams(const Triangle triangle, int width);
    std::vector<Triangle> DelaunayTriangulation(const cv::Rect boundRC, const std::vector<cv::Point>& points);
    void GetTriangulateVertices(std::vector<cv::Point>& Vertices);

    void Release(std::vector<Scene> Scenes,const int &ID);

    void Run();
};


#endif 