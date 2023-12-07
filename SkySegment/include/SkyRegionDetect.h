#ifndef SKYREGIONDETECT_H
#define SKYREGIONDETECT_H

#include <string>
#include <vector>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"

#include "benchmark.h"
#include "datareader.h"
#include "net.h"
using namespace std;
using namespace cv;
//int n = 0;

struct pix
{
    cv::Point p;
    cv::Vec3f rgb;
};

std::vector<float> bias(const std::vector<float> &x, float b);
cv::Mat probability_to_confidence(cv::Mat &mask, const cv::Mat &rgb, float low_thresh, float high_thresh);
cv::Mat downsample2_antialiased(const cv::Mat &X);
cv::Mat self_resize(cv::Mat &X, cv::Size size);
cv::Mat weighted_downsample(cv::Mat &X, const cv::Mat &confidence, int scale, const cv::Size &target_size_input);
std::vector<cv::Mat> weighted_downsample(std::vector<cv::Mat> &M_vec, cv::Mat &confidence, int scale, const cv::Size &target_size_input);
std::vector<cv::Mat> outer_product_images(const cv::Mat &X, const cv::Mat &Y);
cv::Mat solve_ldl3(const std::vector<cv::Mat> &Covar, const cv::Mat &Residual);
cv::Mat smooth_upsample(cv::Mat &X, cv::Size sz);
int mask_refine(std::string img_folder,std::string mask_folder,std::string outname);


class SkySegment{
public:
    SkySegment(const char* params_path, const char* model_path);
    //bool loadModel(std::string params_path, std::string model_path);
    cv::Mat maskExtractor(cv::Mat &dst);
private:
    ncnn::Net skynet;
};
#endif SKYREGIONDETECT_H