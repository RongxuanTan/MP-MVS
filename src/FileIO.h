#ifndef _FILEIO_H_
#define _FILEIO_H_
#include "main.h"


void checkpath(std::string &path);
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
bool DmbVisualize(cv::Mat_<float> &depth);
bool CostVisualize(cv::Mat_<float> &cost);

#endif