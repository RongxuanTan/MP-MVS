#include "PatchMatch.h"
/******************************************************************sky mask*********************************************************************/
// using namespace std;
// using namespace cv;
// //int n = 0;

// struct pix
// {
//     cv::Point p;
//     cv::Vec3f rgb;
// };

// std::vector<float> bias(const std::vector<float> &x, float b = 0.8)
// {
//     /*** bias计算：参考论文公式2 ***/
//     std::vector<float> denom;
//     for (int i = 0; i < x.size(); ++i) {
//         float tmp = x[i]/(((1.f / b) - 2.f) * (1.f - x[i]) + 1.f);
//         denom.push_back(tmp);
//     }
//     return denom;
// }
// cv::Mat probability_to_confidence(cv::Mat &mask, const cv::Mat &rgb, float low_thresh=0.3, float high_thresh=0.5) {
//     /* *
//      * 计算两个高低掩模：设置高低两个阈值，
//      * 低掩模初始化为0，小于低阈值则置1
//      * 高掩模初始化为0，大于高阈值则置1
//      * */
//     cv::Mat low = Mat::zeros(mask.size(), CV_8UC1);
//     cv::Mat high = Mat::zeros(mask.size(), CV_8UC1);

//     int h = mask.rows;
//     int w = mask.cols;
//     std::vector<pix> sky;
//     std::vector<pix> unknow;
//     for (int i = 0; i < h; ++i) {
//         for (int j = 0; j < w; ++j) {
//             if((float)mask.at<float>(i, j) < low_thresh)
//             {
//                 low.at<uchar>(i, j) = 1;
//             }
//             if((float)mask.at<float>(i, j) > high_thresh)
//             {
//                 high.at<uchar>(i, j) = 1;
//             }

//         }
//     }
//     //std::cout << "density: " << sky.size() << " " << unknow.size() << endl;
//     /* Density Estimation Algorithm
//      * 根据论文结果，加入密集度评估后树枝之间会处理得更好，
//      * 目前结果并无明显改善，可能是算法理解有偏差，有待研究
//      * */
// //    int const num = 1024;         //根据论文，为了减小计算量，在天空区域随机取1024个点
// //    list<int>::iterator it_sky;   //迭代器
// //    list<int> sky_index;          //定义链表，保存生成的随机数
// //    int begin, end;               //数字范围
// //    int sum;                      //随机数个数
// //    begin = 0;
// //    end = sky.size()-1;
// //    while (sky_index.size() < num)
// //    {
// //        sky_index.push_back(rand() % (end - begin + 1) + begin);
// //        sky_index.sort();         //排序
// //        sky_index.unique();       //去除相邻的重复随机数中的第一个
// //    }
// //    cout << "sky pix num: " << sky_index.size() << endl;
// //    const float sigma = 0.01f;
// //    const double K = 1.f/sqrt(pow((2.f*CV_PI*sigma*sigma),3));
// //    for (int k = 0; k < unknow.size(); ++k) {
// //        if(k == 0) {
// //            cv::Vec3f unknowrgb = unknow[k].rgb;
// //            double new_pi = 0.f;
// //            for (it_sky = sky_index.begin(); it_sky != sky_index.end(); it_sky++) {
// //                cv::Vec3f skyrgb = sky[*it_sky].rgb;
// //                double sum_tmp = 0.f;
// //                for (int i = 0; i < 3; ++i) {
// //                    sum_tmp = sum_tmp + (unknowrgb[i] - skyrgb[i]) * (unknowrgb[i] - skyrgb[i]);
// //                }
// //                double sim_rgb = exp((-1.f / 2.f * sigma * sigma) * sum_tmp);
// //                sim_rgb = sim_rgb * K;
// //                new_pi = new_pi + sim_rgb;
// //            }
// //            new_pi = new_pi / float(sky_index.size());
// //            // cout << "unknow: " << unknow[k].p << " " << new_pi << endl;
// //            mask.at<float>(unknow[k].p.y, unknow[k].p.x) = new_pi;
// //        }
// //        else
// //        {
// //            continue;
// //        }
// //    }

//     std::vector<float> confidence_low_tmp ;
//     std::vector<float> confidence_high_tmp;
//     /*根据论文公式1：
//      * 1. 低掩模：（l - p）/ l  高掩模：（p - h）/ (1 - h)
//      * 2. 分别计算bias
//      * */
//     for (int i = 0; i < h; ++i) {
//         for (int j = 0; j < w; ++j) {
//             if((int)low.at<uchar>(i, j) == 1)
//             {
//                 confidence_low_tmp.push_back((low_thresh-mask.at<float>(i, j)) / low_thresh);
//             }
//             if((int)high.at<uchar>(i, j) == 1)
//             {
//                 confidence_high_tmp.push_back((mask.at<float>(i, j) - high_thresh) / (1.f - high_thresh));
//             }
//         }
//     }
//     std::vector<float>confidence_low = bias(confidence_low_tmp);
//     std::vector<float>confidence_high = bias(confidence_high_tmp);
//     cv::Mat confidence =  cv::Mat::zeros(mask.size(), CV_32F);
//     float eps = 0.01;
//     vector<float>::iterator iter1 = confidence_low.begin();
//     vector<float>::iterator iter2 = confidence_high.begin();
//     /**参考公式1.计算最后的置信度map**/
//     for (int i = 0; i < h; ++i) {
//         for (int j = 0; j < w; ++j) {
//             if((int)low.at<uchar>(i, j) == 1 )
//             {
//                 confidence.at<float>(i, j) = *iter1;
//                 ++iter1;
//             }
//             else if((int)high.at<uchar>(i, j) == 1)
//             {
//                 confidence.at<float>(i, j) = *iter2;
//                 ++iter2;
//             }
//             else
//             {
//                 continue;
//             }
//             if(confidence.at<float>(i, j) < eps)
//             {
//                 confidence.at<float>(i, j) = eps;
//             }
//         }
//     }
//     return confidence;
// }
// cv::Mat downsample2_antialiased(const cv::Mat &X)
// {
//     /*向下采样方法：卷积 + 金字塔2倍尺度下采样*/
//     /** filter2D和sepFilter2D两种卷积方法都可以 **/
//     Mat dst;
//     Mat kx = (Mat_<float>(4, 1) << 1.f/8.f, 3.f/8.f, 3.f/8.f, 1.f/8.f);
//     Mat ky = (Mat_<float>(1, 4) << 1.f/8.f, 3.f/8.f, 3.f/8.f, 1.f/8.f);
//     Mat kern = (Mat_<float>(3, 3) << 2.f/9.f, 5.f/9.f, 2.f/9.f,
//             2.f/9.f, 5.f/9.f, 2.f/9.f,
//             2.f/9.f, 5.f/9.f, 2.f/9.f);

//     sepFilter2D(X, dst, -1, kx, ky,Point(1,1),0,BORDER_REPLICATE);

//     Mat dowmsample;
//     // opencv降采样
//     float w = (float)dst.cols/ 2.f;
//     float h = (float)dst.rows/ 2.f;
//     pyrDown(dst, dowmsample,Size(round(w), round(h)));
//     return dowmsample;
// }

// cv::Mat self_resize(cv::Mat &X, cv::Size size)
// {
//     int w = X.cols;
//     int h = X.rows;
//     /*若输入图像长宽都大于2倍的目标尺寸，则不断进行向下采样*/
//     while(X.cols >= 2 * size.width && X.rows >= 2 * size.height)
//     {
//         X = downsample2_antialiased(X);
//     }
//     Mat out;
//     /* 线性插值到目标尺寸 */
//     cv::resize(X,out,cv::Size(size.width, size.height),0,0, cv::INTER_LINEAR);
//     return out;
// }

// cv::Mat weighted_downsample(cv::Mat &X, const cv::Mat &confidence, int scale, const cv::Size &target_size_input)
// {

//     Mat XX = X.clone();
//     Mat confi = confidence.clone();
//     cv::Size target_size;
//     int w = XX.cols;
//     int h = XX.rows;
//     if(scale != -1)
//     {
//         target_size = cv::Size((round)((float)w/(float)scale),
//                                (round)((float)h/(float)scale));
//     }
//     else
//     {
//         target_size = target_size_input;
//     }

//     for (int i = 0; i < h; ++i) {
//         for (int j = 0; j < w; ++j) {
//             if(XX.channels() == 3) {
//                 XX.at<cv::Vec3f>(i, j)[0] = confi.at<float>(i, j) * XX.at<cv::Vec3f>(i, j)[0];
//                 XX.at<cv::Vec3f>(i, j)[1] = confi.at<float>(i, j) * XX.at<cv::Vec3f>(i, j)[1];
//                 XX.at<cv::Vec3f>(i, j)[2] = confi.at<float>(i, j) * XX.at<cv::Vec3f>(i, j)[2];
//             }
//             if(XX.channels() == 1)
//             {
//                 XX.at<float>(i, j) = confi.at<float>(i, j) * XX.at<float>(i, j);
//             }
//         }
//     }

//     Mat numerator = self_resize(XX, target_size);

//     Mat denom = self_resize(confi, target_size);

//     for (int i = 0; i < numerator.rows; ++i) {
//         for (int j = 0; j < numerator.cols; ++j) {
//             if(numerator.channels() == 3) {
//                 numerator.at<cv::Vec3f>(i, j)[0] = numerator.at<cv::Vec3f>(i, j)[0] / denom.at<float>(i, j);
//                 numerator.at<cv::Vec3f>(i, j)[1] = numerator.at<cv::Vec3f>(i, j)[1] / denom.at<float>(i, j);
//                 numerator.at<cv::Vec3f>(i, j)[2] = numerator.at<cv::Vec3f>(i, j)[2] / denom.at<float>(i, j);
//             }
//             if(numerator.channels() == 1)
//             {
//                 numerator.at<float>(i, j) = numerator.at<float>(i, j) / denom.at<float>(i, j);
//             }
//         }
//     }
//     return numerator;
// }
// std::vector<cv::Mat> weighted_downsample(std::vector<cv::Mat> &M_vec, cv::Mat &confidence, int scale, const cv::Size &target_size_input)
// {
//     /**传入矩阵通道数为6，分解成2个3通道mat**/
//     Mat confi = confidence.clone();
//     cv::Size target_size;
//     int w = M_vec[0].cols;
//     int h = M_vec[0].rows;
//     if(scale != -1)
//     {
//         target_size = cv::Size((round)((float)w/(float)scale),
//                                (round)((float)h/(float)scale));
//     }
//     else
//     {
//         target_size = target_size_input;
//     }
//     Mat ch1[3], ch2[3];
//     ch1[0] = M_vec[0].clone();
//     ch1[1] = M_vec[1].clone();
//     ch1[2] = M_vec[2].clone();
//     ch2[0] = M_vec[3].clone();
//     ch2[1] = M_vec[4].clone();
//     ch2[2] = M_vec[5].clone();
//     Mat m1,m2;
//     merge(ch1, 3, m1);
//     merge(ch2, 3, m2);
//     /*2个3通道map分别乘以置信度map*/
//     for (int i = 0; i < m1.rows; ++i) {
//         for (int j = 0; j < m1.cols; ++j) {
//             m1.at<cv::Vec3f>(i, j)[0] = confidence.at<float>(i, j) * m1.at<cv::Vec3f>(i, j)[0];
//             m1.at<cv::Vec3f>(i, j)[1] = confidence.at<float>(i, j) * m1.at<cv::Vec3f>(i, j)[1];
//             m1.at<cv::Vec3f>(i, j)[2] = confidence.at<float>(i, j) * m1.at<cv::Vec3f>(i, j)[2];
//             m2.at<cv::Vec3f>(i, j)[0] = confidence.at<float>(i, j) * m2.at<cv::Vec3f>(i, j)[0];
//             m2.at<cv::Vec3f>(i, j)[1] = confidence.at<float>(i, j) * m2.at<cv::Vec3f>(i, j)[1];
//             m2.at<cv::Vec3f>(i, j)[2] = confidence.at<float>(i, j) * m2.at<cv::Vec3f>(i, j)[2];
//         }
//     }

//     Mat m1_re = self_resize(m1,target_size);
//     Mat m2_re = self_resize(m2,target_size);

//     std::vector<Mat> m1_ch,m2_ch;
//     split(m1_re, m1_ch);
//     split(m2_re, m2_ch);
//     std::vector<Mat> chs;
//     chs.push_back(m1_ch[0]);
//     chs.push_back(m1_ch[1]);
//     chs.push_back(m1_ch[2]);
//     chs.push_back(m2_ch[0]);
//     chs.push_back(m2_ch[1]);
//     chs.push_back(m2_ch[2]);

//     Mat conf = confidence.clone();
//     Mat denom = self_resize(conf,target_size);
//     for (int l = 0; l < chs.size(); ++l) {
//         for (int k = 0; k < chs[0].rows; ++k) {
//             for (int i = 0; i < chs[0].cols; ++i) {
//                 chs[l].at<float>(k, i) = chs[l].at<float>(k, i) / denom.at<float>(k, i);
//             }
//         }
//     }
//     return chs;
// }
// std::vector<cv::Mat> outer_product_images(const cv::Mat &X, const cv::Mat &Y)
// {
//     Mat x_input = X.clone();
//     Mat y_input = Y.clone();
//     vector<Mat> channels_x, channels_y;
//     split(X, channels_x);
//     split(Y, channels_y);
//     std::vector<cv::Mat> triu_mat;
//     // 矩阵乘法，3*3 = 9, 取上对角矩阵，参考公式4
//     for (int i = 0; i < channels_x.size(); ++i) {
//         for (int j = 0; j < channels_y.size(); ++j) {
//             Mat mul_mat = channels_x[i].mul(channels_y[j]);
//             if(i <= j)
//             {
//                 triu_mat.push_back(mul_mat);
//             }
//         }
//     }
//     return triu_mat;
// }
// cv::Mat solve_ldl3(const std::vector<Mat> &Covar, const cv::Mat &Residual)
// {
//     // LDL-decomposition 解压缩算法，引用了文献24
//     /*
//      * 参考公式7：
//         d1 = A11  1
//         L_12 = A12 / d1   1
//         d2 = A22 - L_12 * A12
//         L_13 = A13 / d1
//         L_23 = (A23 - L_13 * A12) / d2
//         d3 = A33 - L_13 * A13 - L_23 * L_23 * d2
//         y1 = b1
//         y2 = b2 - L_12 * y1
//         y3 = b3 - L_13 * y1 - L_23 * y2
//         x3 = y3 / d3
//         x2 = y2 / d2 - L_23 * x3
//         x1 = y1 / d1 - L_12 * x2 - L_13 * x3
//      */
//     Mat A11 = Covar[0].clone();
//     Mat A12 = Covar[1].clone();
//     Mat A13 = Covar[2].clone();
//     Mat A22 = Covar[3].clone();
//     Mat A23 = Covar[4].clone();
//     Mat A33 = Covar[5].clone();
//     cv::Mat residual = Residual.clone();
//     std::vector<Mat> b;
//     split(residual, b);
//     int w = A11.cols;
//     int h = A11.rows;
//     cv::Mat L_12 = Mat::zeros(cv::Size (w,h), CV_32F);
//     cv::Mat L_13 = Mat::zeros(cv::Size (w,h), CV_32F);
//     cv::Mat L_23 = Mat::zeros(cv::Size (w,h), CV_32F);
//     cv::Mat d1 = Mat::zeros(cv::Size (w,h), CV_32F);
//     cv::Mat d2 = Mat::zeros(cv::Size (w,h), CV_32F);
//     cv::Mat d3 = Mat::zeros(cv::Size (w,h), CV_32F);
//     cv::Mat y1 = Mat::zeros(cv::Size (w,h), CV_32F);
//     cv::Mat y2 = Mat::zeros(cv::Size (w,h), CV_32F);
//     cv::Mat y3 = Mat::zeros(cv::Size (w,h), CV_32F);
//     cv::Mat x1 = Mat::zeros(cv::Size (w,h), CV_32F);
//     cv::Mat x2 = Mat::zeros(cv::Size (w,h), CV_32F);
//     cv::Mat x3 = Mat::zeros(cv::Size (w,h), CV_32F);
//     for (int i = 0; i < h; ++i) {
//         for (int j = 0; j < w; ++j) {
//             //d1 = A11
//             d1.at<float>(i, j) = A11.at<float>(i, j);
//             //L_12 = A12 / d1
//             L_12.at<float>(i, j) = A12.at<float>(i, j) / d1.at<float>(i, j);
//             //d2 = A22 - L_12 * A12
//             d2.at<float>(i, j) = A22.at<float>(i, j) -  L_12.at<float>(i, j) * A12.at<float>(i, j);
//             //L_13 = A13 / d1
//             L_13.at<float>(i, j) = A13.at<float>(i, j) / d1.at<float>(i, j);
//             //L_23 = (A23 - L_13 * A12) / d2
//             L_23.at<float>(i, j) = (A23.at<float>(i, j) - L_13.at<float>(i, j)*A12.at<float>(i, j)) / d2.at<float>(i, j);
//             //d3 = A33 - L_13 * A13 - L_23 * L_23 * d2
//             d3.at<float>(i, j) = A33.at<float>(i, j) - L_13.at<float>(i, j)*A13.at<float>(i, j) -L_23.at<float>(i, j)*L_23.at<float>(i, j)*d2.at<float>(i, j);
//             //y1 = b1
//             y1.at<float>(i, j) = b[0].at<float>(i, j);
//             //y2 = b2 - L_12 * y1
//             y2.at<float>(i, j) = b[1].at<float>(i, j) - L_12.at<float>(i, j) * y1.at<float>(i, j);
//             //y3 = b3 - L_13 * y1 - L_23 * y2
//             y3.at<float>(i, j) = b[2].at<float>(i, j) - L_13.at<float>(i, j) * y1.at<float>(i, j) - L_23.at<float>(i, j) * y2.at<float>(i, j);
//             //x3 = y3 / d3
//             x3.at<float>(i, j) = y3.at<float>(i, j)/d3.at<float>(i, j);
//             //x2 = y2 / d2 - L_23 * x3
//             x2.at<float>(i, j) = y2.at<float>(i, j)/d2.at<float>(i, j) - L_23.at<float>(i, j) * x3.at<float>(i, j);
//             //x1 = y1 / d1 - L_12 * x2 - L_13 * x3
//             x1.at<float>(i, j) = y1.at<float>(i, j)/d1.at<float>(i, j) - L_12.at<float>(i, j) * x2.at<float>(i, j) - L_13.at<float>(i, j) * x3.at<float>(i, j);
//         }
//     }
//     std::vector<cv::Mat> ldl3_vec;
//     ldl3_vec.push_back(x1);
//     ldl3_vec.push_back(x2);
//     ldl3_vec.push_back(x3);
//     cv::Mat ldl3;
//     merge(ldl3_vec, ldl3);
//     return ldl3;
//     //cv::Mat L_12 = A2 / d1;
// }
// cv::Mat smooth_upsample(cv::Mat &X, cv::Size sz)
// {
//     cv::Mat XX = X.clone();
//     float x[2];
//     float s[2];
//     x[0] = X.cols;
//     x[1] = X.rows;
//     s[0] = sz.width;
//     s[1] = sz.height;

//     float log4ratio_1 = 0.5 * log2(s[0]/x[0]);
//     float log4ratio_2 = 0.5 * log2(s[1]/x[1]);
//     float log4ratio = log4ratio_1 > log4ratio_2? log4ratio_1:log4ratio_2;

//     int num_steps = 1 > round(log4ratio)? 1 : round(log4ratio);

//     float ratio[2];
//     ratio[0] = (float)sz.width / (float)X.cols;
//     ratio[1] = (float)sz.height / (float)X.rows;
//     float ratio_per_step[2];
//     ratio_per_step[0] = x[0] * ratio[0] / (float)num_steps;
//     ratio_per_step[1] = x[1] * ratio[1] / (float)num_steps;

//     for (int i = 1; i < (num_steps+1); ++i) {
//         cv::Size target_shape_for_step = cv::Size (round(ratio_per_step[0] * (float)i), round(ratio_per_step[1] * (float)i));
//         XX = self_resize(XX, target_shape_for_step);
//     }
//     return XX;
// }

// int mask_refine(std::string img_folder,std::string mask_folder,std::string outname) {
//     Mat bgr = imread(img_folder,CV_32F);
//     cvtColor(bgr,bgr,CV_BGR2RGB);
//     Mat src = imread(mask_folder,CV_32F);
//     int W = src.cols;
//     int H = src.rows;
//     cv::resize(bgr,bgr,cv::Size(W,H),cv::INTER_LINEAR);

//     Mat reference = bgr.clone();

//     vector<Mat> channels;
//     split(src, channels);
//     Mat ch1 = channels.at(0);
//     Mat ch2 = channels.at(1);
//     Mat mask, mask2;
//     ch1.convertTo(mask, CV_32F, 1.0/255, 0);
//     reference.convertTo(reference, CV_32FC3, 1.0/255, 0);
//     Mat img = reference.clone();

//     Mat confidence = probability_to_confidence(mask,reference);

//     Mat conf = confidence.clone();
//     Mat refer1 = reference.clone();
//     int kernel = 256;
//     cv::Mat reference_small = weighted_downsample(refer1, confidence, kernel, cv::Size(0,0));

//     int small_h = reference_small.size[0];
//     int small_w = reference_small.size[1];

//     Mat conf1 = confidence.clone();
//     cv::Mat source_small = weighted_downsample(mask, conf1, -1, cv::Size(small_w, small_h));

//     std::vector<cv::Mat> outer_reference = outer_product_images(reference,reference);

//     Mat conf2 = confidence.clone();
//     vector<Mat> Outer_Reference = weighted_downsample(outer_reference, conf2, -1, cv::Size(small_w, small_h));

//     std::vector<cv::Mat> tri_vec_out = outer_product_images(reference_small, reference_small);

// //    //分离 Outer_Reference
//     vector<Mat> covar;
// //    split(Outer_Reference, covar);
//     for (int l = 0; l < tri_vec_out.size(); ++l) {
//         cv::Mat tri = tri_vec_out[l];
//         cv::Mat var_tmp = Mat::zeros(tri.size(), CV_32F);
//         for (int i = 0; i < tri.rows; ++i) {
//             for (int j = 0; j < tri.cols; ++j) {
//                 var_tmp.at<float>(i, j) = Outer_Reference[l].at<float>(i, j) - tri.at<float>(i, j);
//             }
//         }
//         covar.push_back(var_tmp);
//     }

//     cv::Mat ref_src = Mat::zeros(mask.size(), CV_32FC3);
//     for (int i = 0; i < mask.rows; ++i) {
//         for (int j = 0; j < mask.cols; ++j) {
//             ref_src.at<cv::Vec3f>(i, j)[0] = mask.at<float>(i, j) * reference.at<cv::Vec3f>(i, j)[0];
//             ref_src.at<cv::Vec3f>(i, j)[1] = mask.at<float>(i, j) * reference.at<cv::Vec3f>(i, j)[1];
//             ref_src.at<cv::Vec3f>(i, j)[2] = mask.at<float>(i, j) * reference.at<cv::Vec3f>(i, j)[2];
//         }
//     }
//     Mat conf3 = confidence.clone();
//     Mat var = weighted_downsample(ref_src, conf3, -1, cv::Size(small_w, small_h));
// //
// //    /* residual_small = var - reference_small * source_small[..., np.newaxis] */
//     cv::Mat residual_small = Mat::zeros(cv::Size(small_w, small_h), CV_32FC3);
//     for (int i = 0; i < small_h; ++i) {
//         for (int j = 0; j < small_w; ++j) {
//             residual_small.at<cv::Vec3f>(i, j)[0] = var.at<cv::Vec3f>(i, j)[0] - source_small.at<float>(i, j)* reference_small.at<cv::Vec3f>(i, j)[0];
//             residual_small.at<cv::Vec3f>(i, j)[1] = var.at<cv::Vec3f>(i, j)[1] - source_small.at<float>(i, j)* reference_small.at<cv::Vec3f>(i, j)[1];
//             residual_small.at<cv::Vec3f>(i, j)[2] = var.at<cv::Vec3f>(i, j)[2] - source_small.at<float>(i, j)* reference_small.at<cv::Vec3f>(i, j)[2];
//         }
//     }
// //    cv::imwrite("../debug_image/residual_small_c.png",255*residual_small);
//     for (int m = 0; m < covar.size(); ++m) {
//         if(m == 0 || m == 3|| m == 5) {
//             for (int k = 0; k < covar[0].rows; ++k) {
//                 for (int i = 0; i < covar[0].cols; ++i) {
//                     covar[m].at<float>(k, i) = covar[m].at<float>(k, i) + 0.01 * 0.01;
//                 }
//             }
//         }
//     }

//     cv::Mat affine = solve_ldl3(covar, residual_small);

//     cv::Mat residual = Mat::zeros(cv::Size(small_w, small_h), CV_32F);
//     for (int m = 0; m < covar[0].rows; ++m) {
//         for (int i = 0; i < covar[0].cols; ++i) {
//             float r = affine.at<cv::Vec3f>(m, i)[0] * reference_small.at<cv::Vec3f>(m, i)[0];
//             float g = affine.at<cv::Vec3f>(m, i)[1] * reference_small.at<cv::Vec3f>(m, i)[1];
//             float b = affine.at<cv::Vec3f>(m, i)[2] * reference_small.at<cv::Vec3f>(m, i)[2];
//             float sum = r+b+g;
//             residual.at<float>(m, i) = source_small.at<float>(m, i) - sum;
//         }
//     }

//     cv::Mat affine_modify = smooth_upsample(affine, cv::Size(W, H));
//     cv::Mat residual_modify = smooth_upsample(residual, cv::Size(W, H));

//     cv::Mat output = Mat::zeros(cv::Size(W, H), CV_32F);
//     for (int i1 = 0; i1 < output.rows; ++i1) {
//         for (int i = 0; i < output.cols; ++i) {
//             float r = img.at<cv::Vec3f>(i1, i)[0] * affine_modify.at<cv::Vec3f>(i1, i)[0];
//             float g = img.at<cv::Vec3f>(i1, i)[1] * affine_modify.at<cv::Vec3f>(i1, i)[1];
//             float b = img.at<cv::Vec3f>(i1, i)[2] * affine_modify.at<cv::Vec3f>(i1, i)[2];
//             float sum = r+g+b;
//             output.at<float>(i1, i) = sum + residual_modify.at<float>(i1, i);
//             if(output.at<float>(i1, i) > 0.4f)
//             {
//                 output.at<float>(i1, i) = 255.f;
//             }else
//             {
//                 output.at<float>(i1, i) = 0.f;
//             }
//         }
//     }
//     cv::imwrite(outname,output);

//     // bilateralFilter
//     // Mat out_bilf;
//     // bilateralFilter(output, out_bilf, 0, 20, 10);
//     // cv::imwrite("../eval/IMG_20210309_211233_filter.png",out_bilf);

//     //cv::imwrite(outname,255 * output);

//     return 0;
// }

// void GenerateSkyRegionMask(std::vector<Scene> &Scenes,std::string &dense_folder){
//     ncnn::Net skynet;
//     skynet.opt.use_vulkan_compute = true;
//     skynet.load_param("/home/xuan/MP-MVS/src/seg/skysegsmall_sim-opt-fp16.param");
//     skynet.load_model("/home/xuan/MP-MVS/src/seg/skysegsmall_sim-opt-fp16.bin");
//     std::string image_folder = dense_folder + std::string("/images");


//     int n=Scenes.size();
//     for(int i=0;i<n;++i){
//         std::stringstream image_path;
//         image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << Scenes[i].refID << ".jpg";
//         cv::Mat bgr = cv::imread (image_path.str(), 1);
//         cv::Mat dst = bgr.clone();
//         while (dst.rows > 768 && dst.cols >768 ) {
//             pyrDown(dst, dst, cv::Size(dst.cols / 2, dst.rows / 2));
//         }
//         int w = bgr.cols;
//         int h = bgr.rows;
//         ncnn::Mat in = ncnn::Mat::from_pixels_resize(dst.data, ncnn::Mat::PIXEL_BGR2RGB, dst.cols, dst.rows, 384, 384);
//         const float mean_vals[3] =  {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
//         const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
//         in.substract_mean_normalize(mean_vals, norm_vals);
//         ncnn::Extractor ex = skynet.create_extractor();
//         ex.set_light_mode(true);
//         ex.set_num_threads(4);
//         ex.input("input.1", in);
//         ncnn::Mat out;
//         ex.extract("1959", out);
//         cv::Mat opencv_mask(out.h, out.w, CV_32FC1);
//         memcpy((uchar*)opencv_mask.data, out.data, out.w * out.h * sizeof(float));

//         //
//         int max_image_size=3200;
//         if(bgr.cols>max_image_size||bgr.rows>max_image_size)
//         {            
//             const float factor_x = static_cast<float>(max_image_size) / bgr.cols;
//             const float factor_y = static_cast<float>(max_image_size) / bgr.rows;
//             const float factor = std::min(factor_x, factor_y);

//             w = std::round(bgr.cols * factor);
//             h = std::round(bgr.rows * factor);

            
//         }
//         //
//         cv::resize(opencv_mask,opencv_mask,cv::Size(w,h),cv::INTER_LINEAR);

//         std::stringstream result_path;
//         result_path << dense_folder << "/MPMVS" << "/2333_" << std::setw(8) << std::setfill('0') << Scenes[i].refID;        
//         std::string result_folder = result_path.str();
//         //std::cout<<result_folder<<std::endl;
//         mkdir(result_folder.c_str(), 0777);
//         std::string mask_path = result_folder + "/skymask.jpg";
//         std::string refinemask_path = result_folder + "/skymask_refine.jpg";
//         cv::imwrite(mask_path, 255*opencv_mask);

//         mask_refine(image_path.str(),mask_path,refinemask_path);
//     }

// }

/******************************************************************sky mask*********************************************************************/
void checkCudaCall(const cudaError_t error) {
	if (error == cudaSuccess)
		return;
    std::cout<<cudaGetErrorString(error)<<"heppend"<< std::endl;
	exit(EXIT_FAILURE);
}

void GenerateSampleList(const std::string &input_path,std::vector<Scene> &Scenes){
    Scenes.clear();
    std::string cluster_list_path = input_path + std::string("/pair.txt");
    std::ifstream file(cluster_list_path);
    if(!file.is_open()){
        std::cout<<"can not open file in path:   "<<cluster_list_path<<std::endl;
        exit(1);
    }

    int num_images;
    file >> num_images;
    
    for (int i = 0; i < num_images; ++i) {
        Scene scene;
        scene.srcID.clear();
        file >> scene.refID;

        int num_src_images;
        file >> num_src_images;
        for (int j = 0; j < num_src_images; ++j) {
            int id;
            float score;
            file >> id >> score;
            if (score <= 0.0f) {
                continue;
            }
            scene.srcID.push_back(id);
        }
        Scenes.push_back(scene);
    }
}

Camera ReadCamera(const std::string &cam_path)
{
    Camera camera;
    std::ifstream file(cam_path);
    if(!file.is_open()){
        std::cout<<"can not open file in path:   "<<cam_path<<std::endl;
        exit(1);
    }

    std::string line;
    file >> line;

    for (int i = 0; i < 3; ++i) {
        file >> camera.R[3 * i + 0] >> camera.R[3 * i + 1] >> camera.R[3 * i + 2] >> camera.t[i];
    }

    float tmp[4];
    file >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
    file >> line;

    for (int i = 0; i < 3; ++i) {
        file >> camera.K[3 * i + 0] >> camera.K[3 * i + 1] >> camera.K[3 * i + 2];
    }
    camera.C[0] = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    camera.C[1] = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    camera.C[2] = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);

    float depth_num;
    float interval;
    file >> camera.depth_min >> interval >> depth_num >> camera.depth_max;

    return camera;
}

void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc)
{
    std::cout << "store 3D points to ply file" << std::endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath.c_str(), "wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size());
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property float nx\n");
    fprintf(outputPly, "property float ny\n");
    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    //write data
#pragma omp parallel for
    for(size_t i = 0; i < pc.size(); i++) {
        const PointList &p = pc[i];
        float3 X = p.coord;
        const float3 normal = p.normal;
        const float3 color = p.color;
        const char b_color = (int)color.x;
        const char g_color = (int)color.y;
        const char r_color = (int)color.z;

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            fwrite(&X.x,      sizeof(X.x), 1, outputPly);
            fwrite(&X.y,      sizeof(X.y), 1, outputPly);
            fwrite(&X.z,      sizeof(X.z), 1, outputPly);
            fwrite(&normal.x, sizeof(normal.x), 1, outputPly);
            fwrite(&normal.y, sizeof(normal.y), 1, outputPly);
            fwrite(&normal.z, sizeof(normal.z), 1, outputPly);
            fwrite(&r_color,  sizeof(char), 1, outputPly);
            fwrite(&g_color,  sizeof(char), 1, outputPly);
            fwrite(&b_color,  sizeof(char), 1, outputPly);
        }

    }
    fclose(outputPly);
}

float3 Get3DPointonRefCam(const int x, const int y, const float depth, const Camera camera)
{
    float3 pointX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    return pointX;
}

float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera)
{
    float3 pointX;
    float3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    float3 C;
    C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}

float GetAngle( const cv::Vec3f &v1, const cv::Vec3f &v2 )
{
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float angle = acosf(dot_product);
    //if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
    if ( angle != angle )
        return 0.0f;

    return angle;
}

void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

void  RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera)
{
    const int cols = depth.cols;
    const int rows = depth.rows;

    if (cols == src.cols && rows == src.rows) {
        dst = src.clone();
        return;
    }

    const float scale_x = cols / static_cast<float>(src.cols);
    const float scale_y = rows / static_cast<float>(src.rows);

    cv::resize(src, dst, cv::Size(cols,rows), 0, 0, cv::INTER_LINEAR);

    camera.K[0] *= scale_x;
    camera.K[2] *= scale_x;
    camera.K[4] *= scale_y;
    camera.K[5] *= scale_y;
    camera.width = cols;
    camera.height = rows;
}

void RunFusion(std::string &dense_folder, const std::vector<Scene> &Scenes,bool sky_mask){
    size_t num_images = Scenes.size();
    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder = dense_folder + std::string("/cams");

    std::vector<cv::Mat> images;
    std::vector<Camera> cameras;
    std::vector<cv::Mat_<float>> depths;
    std::vector<cv::Mat_<cv::Vec3f>> normals;
    std::vector<cv::Mat> masks;
    images.clear();
    cameras.clear();
    depths.clear();
    normals.clear();
    masks.clear();
    
    std::map<int, int> image_id_2_index;
    //cv::Mat skymask;

    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        image_id_2_index[Scenes[i].refID] = i;
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << Scenes[i].refID << ".jpg";
        cv::Mat_<cv::Vec3b> image = cv::imread (image_path.str(), cv::IMREAD_COLOR);
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << Scenes[i].refID << "_cam.txt";
        Camera camera = ReadCamera(cam_path.str());

        std::stringstream result_path;
        result_path << dense_folder << "/MPMVS" << "/2333_" << std::setw(8) << std::setfill('0') << Scenes[i].refID;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        std::string depth_path = result_folder + suffix;
        std::string normal_path = result_folder + "/normals.dmb";
        cv::Mat_<float> depth;
        cv::Mat_<cv::Vec3f> normal;
        readDepthDmb(depth_path, depth);
        readNormalDmb(normal_path, normal);

        cv::Mat_<cv::Vec3b> scaled_image;
        RescaleImageAndCamera(image, scaled_image, depth, camera);
        images.push_back(scaled_image);
        cameras.push_back(camera);
        depths.push_back(depth);
        normals.push_back(normal);
        cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
        masks.push_back(mask);
    }

    std::vector<PointList> PointCloud;
    PointCloud.clear();

    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        // if(sky_mask){
        //     std::stringstream result_path;
        //     result_path << dense_folder << "/MPMVS" << "/2333_" << std::setw(8) << std::setfill('0') << Scenes[i].refID;
        //     std::string mask_path =result_path.str()+ "/skymask_refine.jpg";
        //     cv::Mat temp=cv::imread(mask_path,1);
            
        //     const int cols = skymask.cols;
        //     const int rows = skymask.rows;

        //     if (cols != depths[i].cols || rows != depths[i].rows) {
        //         cv::resize(temp, skymask, cv::Size(depths[i].cols,depths[i].rows), 0, 0, cv::INTER_LINEAR);
        //     } 
            
        // }
        const int cols = depths[i].cols;
        const int rows = depths[i].rows;
        int num_ngb = Scenes[i].srcID.size();
        std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
        for (int r =0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                // if(sky_mask){
                //     if(skymask.at<float>(r,c)!=0){
                //         masks[i].at<uchar>(r, c) = 1;
                //         continue;
                //     }
                // }
                if (masks[i].at<uchar>(r, c) == 1)
                    continue;
                float ref_depth = depths[i].at<float>(r, c);
                cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);

                if (ref_depth <= 0.0)
                    continue;

                float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
                float3 consistent_Point = PointX;
                cv::Vec3f consistent_normal = ref_normal;
                float consistent_Color[3] = {(float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2]};
                int num_consistent = 0;
                float dynamic_consistency = 0;

                for (int j = 0; j < num_ngb; ++j) {
                    int src_id = image_id_2_index[Scenes[i].srcID[j]];
                    const int src_cols = depths[src_id].cols;
                    const int src_rows = depths[src_id].rows;
                    float2 point;
                    float proj_depth;
                    ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
                    int src_r = int(point.y + 0.5f);
                    int src_c = int(point.x + 0.5f);
                    if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                        if (masks[src_id].at<uchar>(src_r, src_c) == 1)
                            continue;

                        float src_depth = depths[src_id].at<float>(src_r, src_c);
                        cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
                        if (src_depth <= 0.0)
                            continue;

                        float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
                        float2 tmp_pt;
                        ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
                        float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float angle = GetAngle(ref_normal, src_normal);

                        if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
                            used_list[j].x = src_c;
                            used_list[j].y = src_r;

                            float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
                            float cons = exp(-tmp_index);
                            dynamic_consistency += exp(-tmp_index);
                            num_consistent++;
                        }
                    }
                }

                if (num_consistent >= 1 && (dynamic_consistency > 0.3 * num_consistent)) {
                    PointList point3D;
                    point3D.coord = consistent_Point;
                    point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                    point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                    PointCloud.push_back(point3D);

                    for (int j = 0; j < num_ngb; ++j) {
                        if (used_list[j].x == -1)
                            continue;
                        masks[image_id_2_index[Scenes[i].srcID[j]]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
                    }
                }
            }
        }
    }
    std::string ply_path = dense_folder + "/MPMVS/MPMVS_model.ply";
    StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);
}

void ProcessProblem(const std::string &input_folder,const std::string &output_folder, std::vector<Scene> &Scenes, const int ID,bool geom_consistency=false, bool planar_prior=false){
    Scene& scene = Scenes[ID];
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << scene.refID << "..." << std::endl;
    cudaSetDevice(0);
    std::stringstream result_path;
    result_path << output_folder << "/2333_" << std::setw(8) << std::setfill('0') << scene.refID;
    std::string result_folder = result_path.str();
    //std::cout<<result_folder<<std::endl;
    mkdir(result_folder.c_str(), 0777);

    PatchMatchCUDA MP;
    MP.SetFolder(input_folder,output_folder);
    MP.SetGeomConsistencyParams(geom_consistency,planar_prior);
    MP.PatchMatchInit(Scenes,ID);
    MP.AllocatePatchMatch();
    MP.CudaMemInit(Scenes[ID]);
    MP.Run();
    const int width = MP.GetReferenceImageWidth();
    const int height = MP.GetReferenceImageHeight();

    cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    if (planar_prior) {
        std::cout << "Run Planar Prior PatchMatch MVS ";
        MP.SetPlanarPriorParams();
        MP.SetGeomConsistencyParams(false,true);
        const cv::Rect imageRC(0, 0, width, height);
        std::vector<cv::Point> Vertices;
        MP.GetTriangulateVertices(Vertices);
        const auto triangles = MP.DelaunayTriangulation(imageRC, Vertices);
        cv::Mat refImage = MP.GetReferenceImage().clone();
        std::vector<cv::Mat> mbgr(3);
        mbgr[0] = refImage.clone();
        mbgr[1] = refImage.clone();
        mbgr[2] = refImage.clone();
        cv::Mat srcImage;
        cv::merge(mbgr, srcImage);
        for (const auto triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
                cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
            }
        }
        std::string triangulation_path = result_folder + "/triangulation.png";
        cv::imwrite(triangulation_path, srcImage);

        cv::Mat_<float> mask_tri = cv::Mat::zeros(height, width, CV_32FC1);
        std::vector<float4> planeParams_tri;
        planeParams_tri.clear();

        uint32_t idx = 0;
        for (const auto triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
                float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
                float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
                float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));
                //取最长边
                float max_edge_length = std::max(L01, std::max(L02, L12));
                float step = 1.0 / max_edge_length;

                for (float p = 0; p < 1.0; p += step) {
                    for (float q = 0; q < 1.0 - p; q += step) {
                        int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
                        int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
                        mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas
                    }
                }

                // estimate plane parameter
                float4 n4 = MP.GetPriorPlaneParams(triangle,width);
                planeParams_tri.push_back(n4);
                idx++;
            }
        }

        cv::Mat_<float> priordepths = cv::Mat::zeros(height, width, CV_32FC1);
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                if (mask_tri(j, i) > 0) {
                    float d = MP.GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
                    if (d <= MP.GetMaxDepth() && d >= MP.GetMinDepth()) {
                        priordepths(j, i) = d;
                    }
                    else {
                        mask_tri(j, i) = 0;
                    }
                }
            }
        }
        std::string depth_path = result_folder + "/depths_prior.dmb";
        writeDepthDmb(depth_path, priordepths);
        MP.CudaPlanarPriorInitialization(planeParams_tri, mask_tri);
        std::cout << "..." << std::endl;
        MP.Run();
        MP.SetGeomConsistencyParams(geom_consistency,planar_prior);

    }
    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            int idx = row * width + col;
            float4 plane_hypothesis = MP.GetPlaneHypothesis(idx);
            depths(row, col) = plane_hypothesis.w;
            normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            costs(row, col) = MP.GetCost(idx);
            // if(planar_prior)
            // TexCofMap(row, col) = MP.GetTextureCofidence(idx);
        }
    }

    std::string suffix = "/depths.dmb";
    std::string depth_path = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path = result_folder + "/costs.dmb";
    //std::string texcof_path = result_folder + "/TexCofMap.jpg";
    
    writeDepthDmb(depth_path, depths);
    writeNormalDmb(normal_path, normals);
    writeDepthDmb(cost_path, costs);
    // if(planar_prior)
    //     cv::imwrite(texcof_path,TexCofMap);
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << scene.refID << " done!" << std::endl;

    MP.Release(Scenes,ID);
}

float PatchMatchCUDA::GetMinDepth()
{
    return params.depth_min;
}

float PatchMatchCUDA::GetMaxDepth()
{
    return params.depth_max;
}

float PatchMatchCUDA::GetDepthFromPlaneParam(const float4 plane_hypothesis, const int x, const int y)
{
    return -plane_hypothesis.w * cameras[0].K[0] / ((x - cameras[0].K[2]) * plane_hypothesis.x + (cameras[0].K[0] / cameras[0].K[4]) * (y - cameras[0].K[5]) * plane_hypothesis.y + cameras[0].K[0] * plane_hypothesis.z);
}

void PatchMatchCUDA::SetGeomConsistencyParams(bool geom_consistency,bool planar_prior)
{
    params.geom_consistency = geom_consistency;
    if(geom_consistency)
        params.geom_map = planar_prior;
    if(geom_consistency)
        params.max_iterations = 2;
    else
        params.max_iterations = 3;
}

void PatchMatchCUDA::SetPlanarPriorParams()
{
    params.planar_prior = true;
}

float4 PatchMatchCUDA::GetPlaneHypothesis(const int index)
{
    return hostPlaneHypotheses[index];
}

float PatchMatchCUDA::GetCost(const int index)
{
    return hostCosts[index];
}

uchar PatchMatchCUDA::GetTextureCofidence(const int index){
    return hostTexCofMap[index];
}

float PatchMatchCUDA::GetGeomCount(const int index){
    return hostGeomMap[index];
}

int PatchMatchCUDA::GetReferenceImageWidth()
{
    return cameras[0].width;
}

int PatchMatchCUDA::GetReferenceImageHeight()
{
    return cameras[0].height;
}

cv::Mat PatchMatchCUDA::GetReferenceImage(){
    return images[0];
}

void PatchMatchCUDA::SetFolder(const std::string &_input_folder,const std::string &_output_folder){
    input_folder=_input_folder;
    output_folder=_output_folder;
}

float4 PatchMatchCUDA::GetPriorPlaneParams(const Triangle triangle,int width)
{
    cv::Mat A(3, 4, CV_32FC1);
    cv::Mat B(4, 1, CV_32FC1);
    // float3 ptX1 = Get3DPointonRefCam(triangle.pt1.x, triangle.pt1.y, depths(triangle.pt1.y, triangle.pt1.x), cameras[0]);
    // float3 ptX2 = Get3DPointonRefCam(triangle.pt2.x, triangle.pt2.y, depths(triangle.pt2.y, triangle.pt2.x), cameras[0]);
    // float3 ptX3 = Get3DPointonRefCam(triangle.pt3.x, triangle.pt3.y, depths(triangle.pt3.y, triangle.pt3.x), cameras[0]);
    float3 ptX1 = Get3DPointonRefCam(triangle.pt1.x, triangle.pt1.y, GetPlaneHypothesis(triangle.pt1.y * width + triangle.pt1.x).w, cameras[0]);
    float3 ptX2 = Get3DPointonRefCam(triangle.pt2.x, triangle.pt2.y, GetPlaneHypothesis(triangle.pt2.y * width + triangle.pt2.x).w, cameras[0]);
    float3 ptX3 = Get3DPointonRefCam(triangle.pt3.x, triangle.pt3.y, GetPlaneHypothesis(triangle.pt3.y * width + triangle.pt3.x).w, cameras[0]);    
    //计算平面参数
    A.at<float>(0, 0) = ptX1.x;
    A.at<float>(0, 1) = ptX1.y;
    A.at<float>(0, 2) = ptX1.z;
    A.at<float>(0, 3) = 1.0;
    A.at<float>(1, 0) = ptX2.x;
    A.at<float>(1, 1) = ptX2.y;
    A.at<float>(1, 2) = ptX2.z;
    A.at<float>(1, 3) = 1.0;
    A.at<float>(2, 0) = ptX3.x;
    A.at<float>(2, 1) = ptX3.y;
    A.at<float>(2, 2) = ptX3.z;
    A.at<float>(2, 3) = 1.0;
    cv::SVD::solveZ(A, B);
    float4 n4 = make_float4(B.at<float>(0, 0), B.at<float>(1, 0), B.at<float>(2, 0), B.at<float>(3, 0));
    float norm2 = sqrt(pow(n4.x, 2) + pow(n4.y, 2) + pow(n4.z, 2));
    if (n4.w < 0) {
        norm2 *= -1;
    }
    n4.x /= norm2;
    n4.y /= norm2;
    n4.z /= norm2;
    n4.w /= norm2;

    return n4;
}

std::vector<Triangle> PatchMatchCUDA::DelaunayTriangulation(const cv::Rect boundRC, const std::vector<cv::Point>& points){
    if (points.empty()) {
        std::cout<<"No Point to Triangulate!"<<std::endl;
        exit(1);
    }

    std::vector<Triangle> results;

    std::vector<cv::Vec6f> temp_results;
    //创建三角剖分对象
    cv::Subdiv2D subdiv2d(boundRC);
    for (const auto point : points) {//插入三角剖分顶点
        subdiv2d.insert(cv::Point2f((float)point.x, (float)point.y));
    }
    //计算剖分结果
    subdiv2d.getTriangleList(temp_results);

    for (const auto temp_vec : temp_results) {
        cv::Point pt1((int)temp_vec[0], (int)temp_vec[1]);
        cv::Point pt2((int)temp_vec[2], (int)temp_vec[3]);
        cv::Point pt3((int)temp_vec[4], (int)temp_vec[5]);
        results.push_back(Triangle(pt1, pt2, pt3));
    }
    return results;
}

void PatchMatchCUDA::GetTriangulateVertices(std::vector<cv::Point>& Vertices){
    Vertices.clear();
    const int step_size = 5;
    const int width = GetReferenceImageWidth();
    const int height = GetReferenceImageHeight();
    for (int col = 0; col < width; col += step_size) {
        for (int row = 0; row < height; row += step_size) {
            float min_cost = 2.0f;
            float minCosts[3] = {2.0f,2.0f,2.0f};
            std::vector<cv::Point> points(3);
            cv::Point temp_point;
            int c_bound = std::min(width, col + step_size);
            int r_bound = std::min(height, row + step_size);
            for (int c = col; c < c_bound; ++c) {
                for (int r = row; r < r_bound; ++r) {
                    int idx = r * width + c;
                    float cost = GetCost(idx);
                    if(!params.geom_map){
                        if (cost < 0.1f && min_cost > cost) {
                        temp_point = cv::Point(c, r);
                        min_cost = cost;
                        }
                    }else if (cost < 0.16f && ( GetGeomCount(idx)< 0.3f) && cost < minCosts[2]){
                        //
                        minCosts[2] = cost;
                        points[2] = cv::Point(c, r);
                        for(int i = 1; i >= 0; --i){
                            if(minCosts[i] <= minCosts[i+1]) break;
                            
                            float temp = minCosts[i+1];  
                            minCosts[i+1] = minCosts[i];
                            minCosts[i] = temp;
                            
                            cv::Point tp = points[i+1];
                            points[i+1] = points[i];
                            points[i] = tp;
                        }
                        //
                        // if (cost < 0.2f && ( GetGeomCount(idx)< 0.4f) && min_cost > cost) {
                        // temp_point = cv::Point(c, r);
                        // min_cost = cost;
                        // }
                    }
                }
            }
            if(params.geom_map){
                for(int i = 0; i <3; ++i){
                    if(minCosts[i] < 0.2f){
                        Vertices.push_back(points[i]);
                    }else
                        break;
                }
            }else{
                if(min_cost < 0.1f)
                    Vertices.push_back(temp_point);
            }
         }
    }
}

void PatchMatchCUDA::DataInit(){
    images.clear();
    depths.clear();
    cameras.clear();
    cudaImageArrays.clear();
    textureImages.clear();
}

void PatchMatchCUDA::PatchMatchInit(std::vector<Scene> Scenes,const int ID){
    //clear data 
    DataInit();

    Scene& scene = Scenes[ID];
    //set images and camera data
    std::string image_folder = input_folder + std::string("/images");
    std::string cam_folder = input_folder + std::string("/cams");   

    std::stringstream image_path;
    image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << scene.refID << ".jpg";
    cv::Mat_<uint8_t> img_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
    if(img_uint.empty())  //判断是否有数据
    {  
        std::cout<<"Can not read this image !"<<image_path.str()<<std::endl;   
    }
    img_uint.convertTo(scene.image,CV_32FC1);
    images.push_back(scene.image);
    std::stringstream cam_path;
    cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << scene.refID << "_cam.txt";
    Camera cam=ReadCamera(cam_path.str());
    cam.height=scene.image.rows;
    cam.width=scene.image.cols;
    cameras.push_back(cam);

    size_t n = scene.srcID.size();
    for (size_t i = 0; i < n; ++i) {
        Scene& Srcscene = Scenes[scene.srcID[i]];
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << scene.srcID[i] << ".jpg";
        cv::Mat_<uint8_t> img_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
        if(img_uint.empty())  //判断是否有数据
        {   
            std::cout<<"Can not read this image !"<<image_path.str()<<std::endl;  
        }
        img_uint.convertTo(Srcscene.image,CV_32FC1);
        images.push_back(Srcscene.image);
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << scene.srcID[i] << "_cam.txt";
        Camera cam=ReadCamera(cam_path.str());
        cam.height=Srcscene.image.rows;
        cam.width=Srcscene.image.cols;
        cameras.push_back(cam);
    }

    num_img=images.size();
    
    //Adjust image scale
    int max_image_size = scene.max_image_size;
    for (size_t i = 0; i < num_img; ++i) {
        if (images[i].cols <= max_image_size && images[i].rows <= max_image_size) {
            continue;
        }
        if(i==0){
            const float factor_x = static_cast<float>(max_image_size) / images[i].cols;
            const float factor_y = static_cast<float>(max_image_size) / images[i].rows;
            const float factor = std::min(factor_x, factor_y);

            const int new_cols = std::round(images[i].cols * factor);
            const int new_rows = std::round(images[i].rows * factor);

            const float scale_x = new_cols / static_cast<float>(images[i].cols);
            const float scale_y = new_rows / static_cast<float>(images[i].rows);

            cv::Mat_<float> scaled_image_float;
            cv::resize(scene.image, scaled_image_float, cv::Size(new_cols,new_rows), 0, 0, cv::INTER_LINEAR);
            scene.image=scaled_image_float.clone();
            images[i] = scene.image;
        
            cameras[i].K[0] *= scale_x;
            cameras[i].K[2] *= scale_x;
            cameras[i].K[4] *= scale_y;
            cameras[i].K[5] *= scale_y;
            cameras[i].height = scaled_image_float.rows;
            cameras[i].width = scaled_image_float.cols;
            continue;
        }
        const int srcID=scene.srcID[i - 1];
        max_image_size = Scenes[srcID].max_image_size;
        const float factor_x = static_cast<float>(max_image_size) / images[i].cols;
        const float factor_y = static_cast<float>(max_image_size) / images[i].rows;
        const float factor = std::min(factor_x, factor_y);

        const int new_cols = std::round(images[i].cols * factor);
        const int new_rows = std::round(images[i].rows * factor);

        const float scale_x = new_cols / static_cast<float>(images[i].cols);
        const float scale_y = new_rows / static_cast<float>(images[i].rows);

        cv::Mat_<float> scaled_image_float;
        if(Scenes[srcID].image.empty())  //判断是否有数据
        {   
            std::cout<<"src:"<<srcID<<std::endl;
            std::cout<<"Can not read this image !"<<srcID<<std::endl;  
        }
        cv::resize(Scenes[srcID].image, scaled_image_float, cv::Size(new_cols,new_rows), 0, 0, cv::INTER_LINEAR);
        Scenes[srcID].image=scaled_image_float.clone();
        images[i] = Scenes[srcID].image;

        cameras[i].K[0] *= scale_x;
        cameras[i].K[2] *= scale_x;
        cameras[i].K[4] *= scale_y;
        cameras[i].K[5] *= scale_y;
        cameras[i].height = scaled_image_float.rows;
        cameras[i].width = scaled_image_float.cols;
    }
    std::cout<<"Refrencce image "<<scene.refID<<" start compute."<<"There are "<<n<<" source images can be used."<<std::endl;

    //patchmatch params setting
    params.depth_min = cameras[0].depth_min * 0.6f;
    params.depth_max = cameras[0].depth_max * 1.2f;
    std::cout << "Set depth range: " << params.depth_min << " to " << params.depth_max << std::endl;
    params.num_images = (int)images.size();

    if (params.geom_consistency) {
        std::cout<<"Geometry consistency optimize start"<<std::endl;
        depths.clear();
        //read depths
        std::stringstream result_path;
        result_path << input_folder << "/MPMVS" << "/2333_" << std::setw(8) << std::setfill('0') << scene.refID;
        //std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";

        //std::string depth_path = result_folder + suffix;
        // readDepthDmb(depth_path, scene.depth);
        // depths.push_back(scene.depth);

        size_t num_src_images = scene.srcID.size();
        for (size_t i = 0; i < num_src_images; ++i) {
            Scene& Srcscene = Scenes[scene.srcID[i]];
            std::stringstream result_path;
            result_path << input_folder << "/MPMVS"  << "/2333_" << std::setw(8) << std::setfill('0') << scene.srcID[i];
            std::string result_folder = result_path.str();
            std::string depth_path = result_folder + suffix;
            readDepthDmb(depth_path, Srcscene.depth);
            depths.push_back(Srcscene.depth);
        }
    }
    //CUDA
    cudaImageArrays.resize(num_img);
    textureImages.resize(num_img);
    if(params.geom_consistency){
        cudaDepthArrays.resize(num_img-1);
        textureDepths.resize(num_img-1);     
    }
}

void PatchMatchCUDA::AllocatePatchMatch(){
    const size_t wh=cameras[0].width*cameras[0].height;
    checkCudaCall(cudaMalloc((void**)&cudaTextureImages, sizeof(cudaTextureObject_t) * num_img));
    checkCudaCall(cudaMalloc((void**)&cudaCameras, sizeof(Camera) * num_img));
    if(params.geom_consistency){
        checkCudaCall(cudaMalloc((void**)&cudaTextureDepths, sizeof(cudaTextureObject_t) * (num_img-1)));
        hostGeomMap = new float[wh];
        checkCudaCall(cudaMalloc((void**)&cudaGeomMap, sizeof(float) * wh));
        if(params.geom_map){
            // hostGeomMap = new float[wh];
            // checkCudaCall(cudaMalloc((void**)&cudaGeomMap, sizeof(float) * wh));
            hostTexCofMap = new uchar[wh];
            checkCudaCall(cudaMalloc((void**)&cudaTexCofMap, sizeof(uchar) * wh));
        }
    }
    //平面假设
    hostPlaneHypotheses = new float4[wh];
    checkCudaCall(cudaMalloc((void**)&cudaPlaneHypotheses, sizeof(float4) * wh));
    hostCosts = new float[wh];
    checkCudaCall(cudaMalloc((void**)&cudaCosts, sizeof(float) * wh));
    checkCudaCall(cudaMalloc((void**)&cudaRandStates, sizeof(curandState) * wh));
    checkCudaCall(cudaMalloc((void**)&cudaSelectedViews, sizeof(unsigned int) * wh));
}

void PatchMatchCUDA::CudaPlanarPriorInitialization(const std::vector<float4> &PlaneParams, const cv::Mat_<float> &masks){
    hostPriorPlanes = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&cudaPriorPlanes, sizeof(float4) * (cameras[0].height * cameras[0].width));

    hostPlaneMask = new unsigned int[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&cudaPlaneMask, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

    for (int i = 0; i < cameras[0].width; ++i) {
        for (int j = 0; j < cameras[0].height; ++j) {
            int idx = j * cameras[0].width + i;
            hostPlaneMask[idx] = (unsigned int)masks(j, i);
            if (masks(j, i) > 0) {
                hostPriorPlanes[idx] = PlaneParams[masks(j, i) - 1];
            }
        }
    }
    cudaMemcpy(cudaPriorPlanes, hostPriorPlanes, sizeof(float4) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaPlaneMask, hostPlaneMask, sizeof(unsigned int) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
}

void PatchMatchCUDA::CudaMemInit(Scene &scene){
    //分配cuda图像内存空间，创建纹理内存以及纹理绑定
    for(int i=0;i<num_img;++i){
        int rows=images[i].rows;
        int cols=images[i].cols;
        //纹理内存
        //定义纹理具体内容
        const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        //分配GPU内存空间
        checkCudaCall(cudaMallocArray(&cudaImageArrays[i], &channelDesc, cols, rows));
        //(二维)线性内存到2维数组的拷贝
        checkCudaCall(cudaMemcpy2DToArray(cudaImageArrays[i],0,0, images[i].ptr<float>(), images[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice));
        
        //创建资源描述符
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cudaImageArrays[i];

        //创建纹理描述符
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        //纹理访问超出边界时的处理方式
        //cudaAddressModeClamp模式：坐标超出范围返回边缘值、cudaAddressModeBorder模式：坐标超出范围则返回 0、
        //cudaAddressModeWrap 和 cudaAddressModeMirror模式：将图像看成周期函数进行循环访问，并且Mirror为镜像循环 (即 “123123“ 和 “123321”) 
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        // cudaReadModeElementType为默认方式，cudaReadModeNormalizedFloat 将整型归一化为单精度浮点类型 [-1,1] 或 [0,1]。
        texDesc.readMode  = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        checkCudaCall(cudaCreateTextureObject(&textureImages[i], &resDesc, &texDesc, NULL));
        //cudaError_t cudaMemcpy2DToArray (struct cudaArray *  	dst,size_t  wOffset,size_t  hOffset,const void *  src,size_t  spitch,size_t  width,size_t 	height,enum cudaMemcpyKind 	kind)
        
        
    }
    //纹理内存图像
    checkCudaCall(cudaMemcpy(cudaTextureImages, textureImages.data(), sizeof(cudaTextureObject_t)*num_img, cudaMemcpyHostToDevice));
    //相机参数
    checkCudaCall(cudaMemcpy(cudaCameras, cameras.data(), sizeof(Camera)*num_img, cudaMemcpyHostToDevice));
    
    if(params.geom_consistency){
        for(int i=0;i<num_img-1;++i){
            int rows=depths[i].rows;
            int cols=depths[i].cols;
            const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            checkCudaCall(cudaMallocArray(&cudaDepthArrays[i], &channelDesc, cols, rows));
            checkCudaCall(cudaMemcpy2DToArray(cudaDepthArrays[i],0,0, depths[i].ptr<float>(), depths[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice));
            
            //创建资源描述符
            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cudaDepthArrays[i];

            //创建纹理描述符
            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode  = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            checkCudaCall(cudaCreateTextureObject(&textureDepths[i], &resDesc, &texDesc, NULL));  
        }
        checkCudaCall(cudaMemcpy(cudaTextureDepths, textureDepths.data(), sizeof(cudaTextureObject_t)*(num_img-1), cudaMemcpyHostToDevice));
        //read normals and costs
        std::stringstream result_path;
        result_path << input_folder << "/MPMVS" << "/2333_" << std::setw(8) << std::setfill('0') << scene.refID;
        std::string result_folder = result_path.str();
        std::string depth_path = result_folder + "/depths.dmb";
        std::string normal_path = result_folder + "/normals.dmb";
        std::string cost_path = result_folder + "/costs.dmb";
        std::string sacle_path = result_folder + "/bestscale.jpg";
        cv::Mat_<float> ref_depth;
        cv::Mat_<cv::Vec3f> ref_normal;
        cv::Mat_<float> ref_cost;
        readDepthDmb(depth_path, ref_depth);
        readNormalDmb(normal_path, ref_normal);
        readDepthDmb(cost_path, ref_cost);

        if(ref_depth.empty()) 
        {   
            std::cout<<"Can not read this depth image !"<<std::endl; 
            exit(0); 
        }
        
        int width = ref_depth.cols;
        int height = ref_depth.rows;
        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int idx= row * width + col;
                float4 PlaneHypothese;
                PlaneHypothese.x = ref_normal(row, col)[0];
                PlaneHypothese.y = ref_normal(row, col)[1];
                PlaneHypothese.z = ref_normal(row, col)[2];
                PlaneHypothese.w = ref_depth.at<float>(row, col);
                hostPlaneHypotheses[idx] = PlaneHypothese;
                hostCosts[idx] = ref_cost(row, col);
            }
        }
        checkCudaCall(cudaMemcpy(cudaPlaneHypotheses, hostPlaneHypotheses, sizeof(float4)*width*height, cudaMemcpyHostToDevice));
        checkCudaCall(cudaMemcpy(cudaCosts, hostCosts, sizeof(float)*width*height, cudaMemcpyHostToDevice));
    }

}

void PatchMatchCUDA::Release(std::vector<Scene> Scenes,const int &ID)
{
    delete[] hostPlaneHypotheses;
    delete[] hostCosts;

    Scene& scene = Scenes[ID];
    scene.image.release();
    std::vector<int> &srcIDs=scene.srcID;
    for(auto id:srcIDs){
        Scene& Srcscene = Scenes[id];
        Srcscene.image.release();
    }

	for(int i=0;i<num_img;++i) 
    {
		cudaDestroyTextureObject(textureImages[i]);
		cudaFreeArray(cudaImageArrays[i]);
	}
    cudaFree(cudaTextureImages);
    cudaFree(cudaCameras);
    cudaFree(cudaPlaneHypotheses);
    cudaFree(cudaCosts);
    cudaFree(cudaRandStates);
    cudaFree(cudaSelectedViews);

    if(params.planar_prior){
        delete[] hostPriorPlanes;
        delete[] hostPlaneMask;
        cudaFree(cudaPriorPlanes);
        cudaFree(cudaPlaneMask);
    }
    if(params.geom_map){
        // delete[] hostGeomMap;
        // cudaFree(cudaGeomMap);
        delete[] hostTexCofMap;
        cudaFree(cudaTexCofMap);
    }
    if(params.geom_consistency){
        scene.depth.release();
        for(auto id:srcIDs){
            Scene& Srcscene = Scenes[id];
            Srcscene.depth.release();
        }
        for(int i=0;i<num_img-1;++i) 
        {
           cudaDestroyTextureObject(textureDepths[i]);
		   cudaFreeArray(cudaDepthArrays[i]); 
        } 
        delete[] hostGeomMap;
        cudaFree(cudaGeomMap);
        cudaFree(cudaTextureDepths);  
        
	}
}

