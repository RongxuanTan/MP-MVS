#ifndef _UTILITY_H_
#define _UTILITY_H_

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"

#include <dirent.h>
#include <Eigen/Dense>

struct Scene
{
    /* data */
    int refID;
    std::vector<int> srcID;
    cv::Mat image;
    cv::Mat_<float> depth;
    int max_image_size = 3200;
};

struct ConfigParams
{
    std::string input_folder;
    std::string output_folder;

    int geom_iterations;
    bool geom_consistency;
    bool planar_prior;
    bool geomPlanarPrior;
    bool sky_seg;

    bool saveDmb;
    bool saveProirDmb;
    bool saveCostDmb;
    bool saveNormalDmb;

    int MaxSourceImageNum;
    int MaxImageSize;
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
void GetHist(cv::Mat gray);
void ShowHist(cv::Mat &Hist);
int getTop10(cv::Mat Hist);
int getDown10(cv::Mat Hist);
void Colormap2Bgr(cv::Mat &src,cv::Mat &dst,cv::Mat &mask);
bool SaveCost(cv::Mat_<float> &cost, const std::string save_path);
bool SaveDmb(cv::Mat_<float> &depth, const std::string save_path, bool hist_enhance);
void saveDmbAsJpg(ConfigParams &config, size_t num_images, bool hist_enhance);
void SaveNormal(cv::Mat_<cv::Vec3f> &normal, const std::string save_path, float k );

#endif