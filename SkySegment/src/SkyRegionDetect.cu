#include "SkyRegionDetect.h"

__global__ void Pixel_bilateral_filter(const uchar *img, const float *mask, float *result, const int height, const int width){
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= width || p.y >= height) {
        return;
    }
    const int idx = p.y * width + p.x;
    float sigma_spatial = 2.0 * 6.0 * 6.0;
    float sigma_color = 2.0 * 2.0 * 2.0;
    float weight_sum = 0.0f;
    float prob = 0.0f;
    const int half_windows = 18;
    float center_color[3] = {(float)img[3 * idx],(float)img[3 * idx + 1],(float)img[3 * idx + 2]};

    for(int i = -half_windows; i <= half_windows; ++i){
        for(int j = -half_windows; j <= half_windows; ++j) {
            int2 np = make_int2(p.x + i , p.y + j);
            int idx2 = np.y * width + np.x;
            if(np.x<0 || np.x >= width || np.y < 0 || np.y >= height)
                continue;
            float b = (float)img[3 * idx2] - center_color[0];
            float g = (float)img[3 * idx2 + 1] - center_color[1];
            float r = (float)img[3 * idx2 + 2] - center_color[2];
            float dis_color = sqrt( b * b + g * g + r * r);
            float distance = sqrt((float)(i * i + j * j));
            float tmp_weight = exp(-distance / sigma_spatial - dis_color / sigma_color);
            weight_sum += tmp_weight;
            float p = tmp_weight *  mask[idx2];
            prob += p;
        }
    }
    prob = prob / weight_sum;
    result[idx] = prob > 0.6 ? 255 : 0;
}

void bilateral_filter(cv::Mat img, cv::Mat &mask_, cv::Mat& result){
    const int height = img.rows;
    const int width = img.cols;

    cv::Mat mask2;
    cv::resize(mask_,mask2,cv::Size(width,height),cv::INTER_LINEAR);

    uchar *src;
    float *mask, *result_mask;
    cudaMalloc((void**)&src, height * width * 3 * sizeof(uchar));
    cudaMalloc((void**)&mask, height * width * sizeof(float));
    cudaMalloc((void**)&result_mask, height * width * sizeof(float));

    cudaMemcpy(src, (uchar *)(img.data), height * width * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(mask, (float*)(mask2.data), height * width * sizeof(float), cudaMemcpyHostToDevice);

    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);
    const dim3 blockSize(BLOCK_W,BLOCK_H,1);
    const dim3 gridSize((width+BLOCK_W-1)/BLOCK_W,(height+BLOCK_H-1)/BLOCK_H,1);
    Pixel_bilateral_filter<<<gridSize, blockSize>>>(src,mask,result_mask,height,width);
    cudaDeviceSynchronize();

    cv::Mat up_img(cv::Size(width,height),CV_32F);
    cudaMemcpy(up_img.data, result_mask, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    result = up_img.clone();
    cudaFree(src);
    cudaFree(mask);
    cudaFree(result_mask);
}