#ifndef _PATCHMATCH_H_
#define _PATCHMATCH_H_

#include "main.h"
#include "FileIO.h"
#include <cstdarg>
#include "benchmark.h"
#include "datareader.h"
#include "net.h"


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
    bool geom_map=false;
    bool planar_prior = false;
};

struct Scene
{
    /* data */
    int refID;
    std::vector<int> srcID;
    cv::Mat image;
    cv::Mat_<float> depth;
    int max_image_size = 3200;
};

struct Triangle {
    cv::Point pt1, pt2, pt3;
    Triangle (const cv::Point _pt1, const cv::Point _pt2, const cv::Point _pt3) : pt1(_pt1) , pt2(_pt2), pt3(_pt3) {}
};
void GenerateSkyRegionMask(std::vector<Scene> &Scenes,std::string &dense_folder);
void checkCudaCall(const cudaError_t error);
void GenerateSampleList(const std::string &input_path,std::vector<Scene> &Scenes);
Camera ReadCamera(const std::string &cam_path);
void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc);
void ProcessProblem(const std::string &input_folder ,const std::string &output_folder, std::vector<Scene> &Scenes, const int ID,bool geom_consistency, bool planar_prior);
float3 Get3DPointonRefCam(const int x, const int y, const float depth, const Camera camera);
float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera); 
float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2);
void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth);
void  RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera);
void RunFusion(std::string &dense_folder, const std::vector<Scene> &Scenes,bool sky_mask);

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
    uchar *cudaGeomMap;
    uchar *hostGeomMap;

    PatchMatchParams params;

    std::string input_folder;
    std::string output_folder;
public:

    //.cpp
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
    uchar GetGeomCount(const int index);

    float4 GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths);
    std::vector<Triangle> DelaunayTriangulation(const cv::Rect boundRC, const std::vector<cv::Point>& points);
    void GetTriangulateVertices(std::vector<cv::Point>& Vertices);


    void Release(std::vector<Scene> Scenes,const int &ID);

    //.cu
    void Run();
};


#endif 