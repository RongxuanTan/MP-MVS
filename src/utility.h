#ifndef _UTILITY_H_
#define _UTILITY_H_
#include "PatchMatch.h"
#include <dirent.h>
#include <Eigen/Dense>

struct ConfigParams
{
    std::string input_folder;
    std::string output_folder;
    //std::string GT_folder;
    int geom_iterations;
    bool geom_consistency;
    bool planar_prior;
    bool geomPlanarPrior;
    bool sky_seg;
    //bool dmap_eval;
};
void checkpath(std::string &path);
ConfigParams readConfig(const std::string yaml_path);
bool readGT(const std::string file_path, cv::Mat_<float> &depth);
void GetFileNames(std::string path,std::vector<std::string>& filenames);
void GetSubFileNames(std::string path,std::vector<std::string>& filenames);
bool GTVisualize(cv::Mat_<float> &depth);
std::vector<double> DmapEval(const std::string data_folder,const std::string GT_folder,const std::string method_name ,const std::string depth_name,float error);
std::vector<double> ColmapEval(const std::string data_folder,const std::string GT_folder,float error);
bool readColmapDmap(const std::string file_path, cv::Mat_<float> &depth);
bool readDepthDmb(const std::string file_path, cv::Mat_<float> &depth);
int writeDepthDmb(const std::string file_path, const cv::Mat_<float> &depth);
bool readNormalDmb (const std::string file_path, cv::Mat_<cv::Vec3f> &normal);
int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> &normal);
void NormalVisualize(float4 *cudaPlaneHypotheses,const int width,const int height);
void GetHist(cv::Mat gray);
void ShowHist(cv::Mat &Hist);
int getTop10(cv::Mat Hist);
int getDown10(cv::Mat Hist);
void Colormap2Bgr(cv::Mat &src,cv::Mat &dst,cv::Mat &mask);
bool DmbVisualize(cv::Mat_<float> &depth,const std::string name);
bool CostVisualize(cv::Mat_<float> &cost);

#endif