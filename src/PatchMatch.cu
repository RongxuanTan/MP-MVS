#include "PatchMatch.h"


__device__ int Point2Idx(const int2& p, int width) 
{
	return p.y * width + p.x;
}

__device__ float Vec3DotVec3(const float4 vec1, const float4 vec2)
{
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__device__  void sort_small(float *d, const int n)
{
    int j;
    for (int i = 1; i < n; i++) {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j-1]; j--)
            d[j] = d[j-1];
        d[j] = tmp;
    }
}

__device__  void setBit(unsigned int &input, const unsigned int n)
{
    input |= (unsigned int)(1 << n);
}

__device__  int isSet(unsigned int input, const unsigned int n)
{
    return (input >> n) & 1;
}

__device__ void Mat33DotVec3(const float mat[9], const float4 vec, float4 *result)
{
  result->x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
  result->y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
  result->z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
}

__device__ void TransformPDFToCDF(float* probs, const int num_probs)
{
    float prob_sum = 0.0f;
    for (int i = 0; i < num_probs; ++i) {
        prob_sum += probs[i];
    }
    const float inv_prob_sum = 1.0f / prob_sum;

    float cum_prob = 0.0f;
    for (int i = 0; i < num_probs; ++i) {
        cum_prob += probs[i] * inv_prob_sum;
        probs[i] = cum_prob;
    }
    probs[num_probs-1] = 1.f;
}

__device__ int FindMinCostIndex(const float *costs, const int n)
{
    float min_cost = costs[0];
    int min_cost_idx = 0;
    for (int idx = 1; idx < n; ++idx) {
        if (costs[idx] <= min_cost) {
            min_cost = costs[idx];
            min_cost_idx = idx;
        }
    }
    return min_cost_idx;
}

__device__ int FindMaxCostIndex(const float *costs, const int n)
{
    float max_cost = costs[0];
    int max_cost_idx = 0;
    for (int idx = 1; idx < n; ++idx) {
        if (costs[idx] >= max_cost) {
            max_cost = costs[idx];
            max_cost_idx = idx;
        }
    }
    return max_cost_idx;
}

__device__ float ComputeDepthfromPlaneHypothesis(const Camera camera, const float4 PlaneHypothesis, const int2 p)
{
    return -PlaneHypothesis.w * camera.K[0] / ((p.x - camera.K[2]) * PlaneHypothesis.x + (camera.K[0] / camera.K[4]) * (p.y - camera.K[5]) * PlaneHypothesis.y + camera.K[0] * PlaneHypothesis.z);
}

__device__ float4 TransformNormal(const Camera camera, float4 plane_hypothesis)
{
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[3] * plane_hypothesis.y + camera.R[6] * plane_hypothesis.z;
    transformed_normal.y = camera.R[1] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[7] * plane_hypothesis.z;
    transformed_normal.z = camera.R[2] * plane_hypothesis.x + camera.R[5] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

__device__ void TexProcessPixel(cudaTextureObject_t *images,uchar *texMap,const int2& p,int width,int height, const PatchMatchParams params)
{
	if (p.x >= width || p.y >= height)
		return;
    float sum_ref=0.0f;
    float sum_ref2=0.0f;
    const int cows_n=(2*params.nSizeHalfWindow)/params.nSizeStep+1;
    const float inv_nm=1/(float)(cows_n*cows_n);
    for (int i = -params.nSizeHalfWindow; i <= params.nSizeHalfWindow; i += params.nSizeStep) {
        for (int j = -params.nSizeHalfWindow; j <= params.nSizeHalfWindow; j += params.nSizeStep) {
            const int2 pt=make_int2(p.x+i,p.y+j);
            const float refPix=tex2D<float>(images[0],pt.x+ 0.5f,pt.y+ 0.5f);
            sum_ref+=refPix;
            sum_ref2+=refPix*refPix;

        }
    }
    const float var_ref=std::abs(sum_ref2-sum_ref*sum_ref*inv_nm)*inv_nm;
    const int ind=Point2Idx(p,width);
    uchar cost=((uchar)var_ref);
    cost=cost<255?cost:255;
    if(cost>texMap[ind])
        texMap[ind]=cost;
    // if(var_ref>100.0f)
    //     texMap[ind]=(uchar)(200);
}


__global__ void TextureConfMap(cudaTextureObject_t *images,uchar *texMap,int width,int height, const PatchMatchParams params){
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    TexProcessPixel(images,texMap,p,width,height,params);
}
//no problem
__device__ void GetPointI2C(const Camera camera, const int2 p, const float depth, float *X)
{
    X[0] = depth * (p.x - camera.K[2]) / camera.K[0];
    X[1] = depth * (p.y - camera.K[5]) / camera.K[4];
    X[2] = depth;
}
//no problem
__device__ float GetPlane2Origin(const Camera camera, const int2 p, const float depth, const float4 normal)
{
    float X[3];
    GetPointI2C(camera, p, depth, X);
    return -(normal.x * X[0] + normal.y * X[1] + normal.z * X[2]);
}

//no problem 
__device__ float4 GetViewDirection(const Camera camera, const int2 p){
    float4 view_direction;
    view_direction.x = (p.x - camera.K[2]) / camera.K[0];
    view_direction.y = (p.y - camera.K[5]) / camera.K[4];
    view_direction.z = 1;
    view_direction.w = 0;
    return view_direction;
}

__device__ void NormalizeVec3 (float4 *vec)
{
    const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
    const float inverse_sqrt = rsqrtf (normSquared);//平方根倒数
    vec->x *= inverse_sqrt;
    vec->y *= inverse_sqrt;
    vec->z *= inverse_sqrt;
}
//no problem 
__device__ float4 GenerateRandomNormal(const Camera camera, const int2 p, curandState *randState){
    float q1, q2, s;
	do {
		q1 = 2.f * curand_uniform(randState) - 1.f;
		q2 = 2.f * curand_uniform(randState) - 1.f;
		s = q1 * q1 + q2 * q2;
	} while (s >= 1.f);
	const float sq = sqrt(1.f - s);
    float4 normal;
    normal.x = 2.0f * q1 * sq;
    normal.y = 2.0f * q2 * sq;
    normal.z = 1.0f - 2.0f * s;
    normal.w = 0;
    const float4 view_direction = GetViewDirection(camera, p);
    float dot_product = normal.x * view_direction.x + normal.y * view_direction.y + normal.z * view_direction.z;
    if (dot_product > 0.0f) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }
    NormalizeVec3(&normal);
    return normal;
}
//no problem 
__device__ float4 GenerateRandomPlaneHypothesis(const Camera camera, const int2 p, curandState *randState, const float depth_min, const float depth_max){
    float4 PlaneHypothesis = GenerateRandomNormal(camera, p, randState);//随机生成法向量
    float depth = curand_uniform(randState) * (depth_max - depth_min) + depth_min;//随机生成深度
    PlaneHypothesis.w = GetPlane2Origin(camera, p, depth, PlaneHypothesis);//计算平面到相机原点距离
    return PlaneHypothesis;
}

__device__ void ComputeHomography(const Camera ref_camera, const Camera src_camera, const float4 PlaneHypotheses, float *H)
{
    float R_relative[9];
    float C_relative[3];
    float t_relative[3];
    R_relative[0] = src_camera.R[0] * ref_camera.R[0] + src_camera.R[1] * ref_camera.R[1] + src_camera.R[2] *ref_camera.R[2];
    R_relative[1] = src_camera.R[0] * ref_camera.R[3] + src_camera.R[1] * ref_camera.R[4] + src_camera.R[2] *ref_camera.R[5];
    R_relative[2] = src_camera.R[0] * ref_camera.R[6] + src_camera.R[1] * ref_camera.R[7] + src_camera.R[2] *ref_camera.R[8];
    R_relative[3] = src_camera.R[3] * ref_camera.R[0] + src_camera.R[4] * ref_camera.R[1] + src_camera.R[5] *ref_camera.R[2];
    R_relative[4] = src_camera.R[3] * ref_camera.R[3] + src_camera.R[4] * ref_camera.R[4] + src_camera.R[5] *ref_camera.R[5];
    R_relative[5] = src_camera.R[3] * ref_camera.R[6] + src_camera.R[4] * ref_camera.R[7] + src_camera.R[5] *ref_camera.R[8];
    R_relative[6] = src_camera.R[6] * ref_camera.R[0] + src_camera.R[7] * ref_camera.R[1] + src_camera.R[8] *ref_camera.R[2];
    R_relative[7] = src_camera.R[6] * ref_camera.R[3] + src_camera.R[7] * ref_camera.R[4] + src_camera.R[8] *ref_camera.R[5];
    R_relative[8] = src_camera.R[6] * ref_camera.R[6] + src_camera.R[7] * ref_camera.R[7] + src_camera.R[8] *ref_camera.R[8];
    C_relative[0] = (ref_camera.C[0] - src_camera.C[0]);
    C_relative[1] = (ref_camera.C[1] - src_camera.C[1]);
    C_relative[2] = (ref_camera.C[2] - src_camera.C[2]);   
    t_relative[0] = src_camera.R[0] * C_relative[0] + src_camera.R[1] * C_relative[1] + src_camera.R[2] * C_relative[2];
    t_relative[1] = src_camera.R[3] * C_relative[0] + src_camera.R[4] * C_relative[1] + src_camera.R[5] * C_relative[2];
    t_relative[2] = src_camera.R[6] * C_relative[0] + src_camera.R[7] * C_relative[1] + src_camera.R[8] * C_relative[2];

    H[0] = R_relative[0] - t_relative[0] * PlaneHypotheses.x / PlaneHypotheses.w;
    H[1] = R_relative[1] - t_relative[0] * PlaneHypotheses.y / PlaneHypotheses.w;
    H[2] = R_relative[2] - t_relative[0] * PlaneHypotheses.z / PlaneHypotheses.w;
    H[3] = R_relative[3] - t_relative[1] * PlaneHypotheses.x / PlaneHypotheses.w;
    H[4] = R_relative[4] - t_relative[1] * PlaneHypotheses.y / PlaneHypotheses.w;
    H[5] = R_relative[5] - t_relative[1] * PlaneHypotheses.z / PlaneHypotheses.w;
    H[6] = R_relative[6] - t_relative[2] * PlaneHypotheses.x / PlaneHypotheses.w;
    H[7] = R_relative[7] - t_relative[2] * PlaneHypotheses.y / PlaneHypotheses.w;
    H[8] = R_relative[8] - t_relative[2] * PlaneHypotheses.z / PlaneHypotheses.w;

    float tmp[9];
    tmp[0] = H[0] / ref_camera.K[0];
    tmp[1] = H[1] / ref_camera.K[4];
    tmp[2] = -H[0] * ref_camera.K[2] / ref_camera.K[0] - H[1] * ref_camera.K[5] / ref_camera.K[4] + H[2];
    tmp[3] = H[3] / ref_camera.K[0];
    tmp[4] = H[4] / ref_camera.K[4];
    tmp[5] = -H[3] * ref_camera.K[2] / ref_camera.K[0] - H[4] * ref_camera.K[5] / ref_camera.K[4] + H[5];
    tmp[6] = H[6] / ref_camera.K[0];
    tmp[7] = H[7] / ref_camera.K[4];
    tmp[8] = -H[6] * ref_camera.K[2] / ref_camera.K[0] - H[7] * ref_camera.K[5] / ref_camera.K[4] + H[8];

    H[0] = src_camera.K[0] * tmp[0] + src_camera.K[2] * tmp[6];
    H[1] = src_camera.K[0] * tmp[1] + src_camera.K[2] * tmp[7];
    H[2] = src_camera.K[0] * tmp[2] + src_camera.K[2] * tmp[8];
    H[3] = src_camera.K[4] * tmp[3] + src_camera.K[5] * tmp[6];
    H[4] = src_camera.K[4] * tmp[4] + src_camera.K[5] * tmp[7];
    H[5] = src_camera.K[4] * tmp[5] + src_camera.K[5] * tmp[8];
    H[6] = src_camera.K[8] * tmp[6];
    H[7] = src_camera.K[8] * tmp[7];
    H[8] = src_camera.K[8] * tmp[8];
}

__device__ float2 ComputeCorrespondingPoint(const float *H, const int2 p)
{
    float3 pt;
    pt.x = H[0] * p.x + H[1] * p.y + H[2];
    pt.y = H[3] * p.x + H[4] * p.y + H[5];
    pt.z = H[6] * p.x + H[7] * p.y + H[8];
    return make_float2(pt.x / pt.z, pt.y / pt.z);
}

__device__ float2 ComputeCorrespondingPointRow(const float *H, const float2 p)
{
    float3 pt;
    pt.x = p.x + H[0];
    pt.y = p.y + H[3];
    pt.z = 1 + H[6];
    return make_float2(pt.x / pt.z, pt.y / pt.z);
}

__device__ float2 ComputeCorrespondingPointCol(const float *H, const float2 p)
{
    float3 pt;
    pt.x = p.x + H[1];
    pt.y = p.y + H[4];
    pt.z = 1 + H[7];
    return make_float2(pt.x / pt.z, pt.y / pt.z);
}

__device__ float4 TransformNormal2RefCam(const Camera camera, float4 plane_hypothesis)
{
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[1] * plane_hypothesis.y + camera.R[2] * plane_hypothesis.z;
    transformed_normal.y = camera.R[3] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[5] * plane_hypothesis.z;
    transformed_normal.z = camera.R[6] * plane_hypothesis.x + camera.R[7] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

__device__ float ComputeBilateralWeight(const float x_dist, const float y_dist, const float pix, const float center_pix, const float sigma_spatial, const float sigma_color)
{
    const float spatial_dist = sqrt(x_dist * x_dist + y_dist * y_dist);
    const float color_dist = fabs(pix - center_pix);
    return exp(-spatial_dist / (2.0f * sigma_spatial* sigma_spatial) - color_dist / (2.0f * sigma_color * sigma_color));
}

__device__ float ComputeBilateralNCC(const cudaTextureObject_t ref_image, const Camera ref_camera, const cudaTextureObject_t src_image, const Camera src_camera, const int2 p, const float4 PlaneHypotheses, const PatchMatchParams params,const int scale)
{
    // const float cost_max = 2.0f;
    // float H[9];
    // ComputeHomography(ref_camera, src_camera, PlaneHypotheses, H);
    // float2 pt = ComputeCorrespondingPoint(H, p);
    // if (pt.x >= src_camera.width || pt.x < 0.0f || pt.y >= src_camera.height || pt.y < 0.0f) {
    //     return cost_max;
    // }
    // int2 temp = make_int2(p.x-params.nSizeHalfWindow,p.y-params.nSizeHalfWindow);
    // float2 x = ComputeCorrespondingPoint(H, temp);
    // float2 basex(x);
    // for(int i=0;i<9;++i){
    //     H[i]=H[i]*(float)params.nSizeStep;
    // }

    const float cost_max = 2.0f;
    int nSizeStep=2;
    for(int i=0;i<scale;++i){
        nSizeStep=nSizeStep*2;
    }
    int radius = 5*nSizeStep/2;

    float H[9];
    ComputeHomography(ref_camera, src_camera, PlaneHypotheses, H);
    float2 pt = ComputeCorrespondingPoint(H, p);
    if (pt.x >= src_camera.width || pt.x < 0.0f || pt.y >= src_camera.height || pt.y < 0.0f) {
        return cost_max;
    }

    float cost = 0.0f;
    {
        float sum_ref = 0.0f;
        float sum_ref_ref = 0.0f;
        float sum_src = 0.0f;
        float sum_src_src = 0.0f;
        float sum_ref_src = 0.0f;
        float bilateral_weight_sum = 0.0f;
        const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);

        for (int i = -radius; i < radius + 1; i += nSizeStep) {
            float sum_ref_row = 0.0f;
            float sum_src_row = 0.0f;
            float sum_ref_ref_row = 0.0f;
            float sum_src_src_row = 0.0f;
            float sum_ref_src_row = 0.0f;
            float bilateral_weight_sum_row = 0.0f;

            for (int j = -radius; j < radius + 1; j += nSizeStep) {
                const int2 ref_pt = make_int2(p.x + i, p.y + j);
                const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                float2 src_pt = ComputeCorrespondingPoint(H, ref_pt);
                const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);

                float weight = ComputeBilateralWeight(i, j, ref_pix, ref_center_pix, params.sigma_spatial, params.sigma_color);

                sum_ref_row += weight * ref_pix;
                sum_ref_ref_row += weight * ref_pix * ref_pix;
                sum_src_row += weight * src_pix;
                sum_src_src_row += weight * src_pix * src_pix;
                sum_ref_src_row += weight * ref_pix * src_pix;
                bilateral_weight_sum_row += weight;
            }

            sum_ref += sum_ref_row;
            sum_ref_ref += sum_ref_ref_row;
            sum_src += sum_src_row;
            sum_src_src += sum_src_src_row;
            sum_ref_src += sum_ref_src_row;
            bilateral_weight_sum += bilateral_weight_sum_row;
        }
        const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
        sum_ref *= inv_bilateral_weight_sum;
        sum_ref_ref *= inv_bilateral_weight_sum;
        sum_src *= inv_bilateral_weight_sum;
        sum_src_src *= inv_bilateral_weight_sum;
        sum_ref_src *= inv_bilateral_weight_sum;

        const float var_ref = sum_ref_ref - sum_ref * sum_ref;
        const float var_src = sum_src_src - sum_src * sum_src;

        const float kMinVar = 1e-5f;
        if (var_ref < kMinVar || var_src < kMinVar) {
            return cost = cost_max;
        } else {
            const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
            const float var_ref_src = sqrt(var_ref * var_src);
            return cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
        }
    }

    // {
    //     float sum_ref = 0.0f;
    //     float sum_ref_ref = 0.0f;
    //     float sum_src = 0.0f;
    //     float sum_src_src = 0.0f;
    //     float sum_ref_src = 0.0f;
    //     float bilateral_weight_sum = 0.0f;
    //     const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);

    //     for (int i = -params.nSizeHalfWindow; i <= params.nSizeHalfWindow; i += params.nSizeStep) {
    //         for (int j = -params.nSizeHalfWindow; j <= params.nSizeHalfWindow; j += params.nSizeStep) {
    //             const int2 ref_pt = make_int2(p.x + j, p.y + i);
    //             const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
    //             const float2 src_pt = x;
    //             const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);
    //             float weight = ComputeBilateralWeight(j, i, ref_pix, ref_center_pix, params.sigma_spatial, params.sigma_color);
    //             const float weightRefPix = weight * ref_pix;
	// 		    const float weightSrcPix = weight * src_pix;
    //             sum_ref += weightRefPix;
    //             sum_ref_ref += weightRefPix * ref_pix;
    //             sum_src += weightSrcPix;
    //             sum_src_src += weightSrcPix * src_pix;
    //             sum_ref_src += weightRefPix* src_pix;
    //             bilateral_weight_sum += weight;
    //             x=ComputeCorrespondingPointRow(H, x);
    //         }
    //         basex=ComputeCorrespondingPointCol(H, basex);
    //         x=basex;
    //     }

    //     const float var_ref = sum_ref_ref * bilateral_weight_sum - sum_ref * sum_ref;
    //     const float var_src = sum_src_src * bilateral_weight_sum - sum_src * sum_src;

    //     const float kMinVar = 1e-5f;
    //     if (var_ref < kMinVar || var_src < kMinVar) {
    //         return cost_max;
    //     } else {
    //         const float covar_src_ref = sum_ref_src * bilateral_weight_sum - sum_ref * sum_src;
    //         const float var_ref_src = sqrt(var_ref * var_src);
    //         return max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
    //     }
    // }
}

__device__ float4 GeneratePerturbedNormal(const Camera camera, const int2 p, const float4 normal, curandState *rand_state, const float perturbation)
{
    float4 view_direction = GetViewDirection(camera, p);

    const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

    const float sin_a1 = sin(a1);
    const float sin_a2 = sin(a2);
    const float sin_a3 = sin(a3);
    const float cos_a1 = cos(a1);
    const float cos_a2 = cos(a2);
    const float cos_a3 = cos(a3);

    float R[9];
    R[0] = cos_a2 * cos_a3;
    R[1] = cos_a3 * sin_a1 * sin_a2 - cos_a1 * sin_a3;
    R[2] = sin_a1 * sin_a3 + cos_a1 * cos_a3 * sin_a2;
    R[3] = cos_a2 * sin_a3;
    R[4] = cos_a1 * cos_a3 + sin_a1 * sin_a2 * sin_a3;
    R[5] = cos_a1 * sin_a2 * sin_a3 - cos_a3 * sin_a1;
    R[6] = -sin_a2;
    R[7] = cos_a2 * sin_a1;
    R[8] = cos_a1 * cos_a2;

    float4 normal_perturbed;
    Mat33DotVec3(R, normal, &normal_perturbed);

    if (Vec3DotVec3(normal_perturbed, view_direction) >= 0.0f) {
        return normal;
    }

    NormalizeVec3(&normal_perturbed);
    return normal_perturbed;
}

__device__ float ComputeMultiViewInitialCostandSelectedViews(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 PlaneHypotheses, unsigned int *selected_views, const PatchMatchParams params,const int scale)
{
    float cost_max = 2.0f;
    float cost_vector[32] = {2.0f};
    float cost_vector_copy[32] = {2.0f};
    int cost_count = 0;
    int num_valid_views = 0;

    for (int i = 1; i < params.num_images; ++i) {
        float c = ComputeBilateralNCC(images[0], cameras[0], images[i], cameras[i], p, PlaneHypotheses, params,scale);
        cost_vector[i - 1] = c;
        cost_vector_copy[i - 1] = c;
        cost_count++;
        if (c < cost_max) {
            num_valid_views++;
        }
    }

    sort_small(cost_vector, cost_count);
    *selected_views = 0;

    int top_k = min(num_valid_views, params.top_k);
    if (top_k > 0) {
        float cost = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            cost += cost_vector[i];
        }
        float cost_threshold = cost_vector[top_k - 1];
        for (int i = 0; i < params.num_images - 1; ++i) {
            if (cost_vector_copy[i] <= cost_threshold) {
                setBit(*selected_views, i);
            }
        }
        return cost / top_k;
    } else {
        return cost_max;
    }
}
//no problem 
__global__ void InitializeScore(const cudaTextureObject_t* images, Camera* cameras, float4* PlaneHypotheses, float* costs, curandState* randStates, unsigned int *selected_views, float4 *PriorPlanes, unsigned int *PlaneMask, const PatchMatchParams params,const int scale){
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    int width = cameras[0].width;
    int height = cameras[0].height;

    if (p.x >= width || p.y >= height) {
        return;
    }

    const int idx = Point2Idx(p, width);
	curand_init(clock64(), p.y, p.x, &randStates[idx]);

    //随即生成平面假设
    if(!params.geom_consistency&&!params.planar_prior){
        PlaneHypotheses[idx]=GenerateRandomPlaneHypothesis(cameras[0], p, &randStates[idx], params.depth_min, params.depth_max);
        costs[idx] = ComputeMultiViewInitialCostandSelectedViews(images, cameras, p, PlaneHypotheses[idx], &selected_views[idx], params,scale);
    }else if(params.planar_prior && PlaneMask[idx] > 0 && costs[idx] >= 0.1f){
        float perturbation = 0.02f;
        float4 plane_hypothesis = PriorPlanes[idx];
        float depth_perturbed = plane_hypothesis.w;
        const float depth_min_perturbed = (1 - 3 * perturbation) * depth_perturbed;
        const float depth_max_perturbed = (1 + 3 * perturbation) * depth_perturbed;
        depth_perturbed = curand_uniform(&randStates[idx]) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;
        float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(cameras[0], p, plane_hypothesis, &randStates[idx], 3 * perturbation * M_PI);
        plane_hypothesis_perturbed.w = depth_perturbed;
        PlaneHypotheses[idx] = plane_hypothesis_perturbed;
        costs[idx] = ComputeMultiViewInitialCostandSelectedViews(images, cameras, p, PlaneHypotheses[idx], &selected_views[idx], params,scale);
    }else
    {
        float4 plane_hypothesis=PlaneHypotheses[idx];
        plane_hypothesis = TransformNormal2RefCam(cameras[0], plane_hypothesis);
        float depth = plane_hypothesis.w;
        plane_hypothesis.w = GetPlane2Origin(cameras[0], p, depth, plane_hypothesis);
        PlaneHypotheses[idx] = plane_hypothesis;
        costs[idx] = ComputeMultiViewInitialCostandSelectedViews(images, cameras, p, PlaneHypotheses[idx], &selected_views[idx], params,scale);
    }
    
}

__device__ void ComputeMultiViewCostVector(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 PlaneHypothesis, float *cost_vector, const PatchMatchParams params,const int scale)
{
    for (int i = 1; i < params.num_images; ++i) {
        cost_vector[i - 1] = ComputeBilateralNCC(images[0], cameras[0], images[i], cameras[i], p, PlaneHypothesis, params, scale);
    }
}

__device__ float3 BackProjectPoint2W(const float x, const float y, const float depth, const Camera camera)
{
    float3 pointX;
    float3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Transformation
    // pointX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z + camera.C[0];
    // pointX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z + camera.C[1];
    // pointX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z + camera.C[2];
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;
    pointX.x = tmpX.x + camera.C[0];
    pointX.y = tmpX.y + camera.C[1];
    pointX.z = tmpX.z + camera.C[2];
    return pointX;

}

__device__ void ProjectPoint(const float3 PointX, const Camera camera, float2 &point)
{
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    const float depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;    
}

__device__ float ComputeGeomConsistencyCost(const cudaTextureObject_t depth_image, const Camera ref_camera, const Camera src_camera, const float4 PlaneHypothesis, const int2 p)
{
    const float max_cost = 3.0f;

    float depth = ComputeDepthfromPlaneHypothesis(ref_camera, PlaneHypothesis, p);
    float3 forward_point = BackProjectPoint2W(p.x, p.y, depth, ref_camera);

    float2 src_pt;
    ProjectPoint(forward_point, src_camera, src_pt);
    const float src_depth = tex2D<float>(depth_image,  (int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);

    if (src_depth == 0.0f) {
        return max_cost;
    }

    float3 src_3D_pt = BackProjectPoint2W(src_pt.x, src_pt.y, src_depth, src_camera);

    float2 backward_point;
    ProjectPoint(src_3D_pt, ref_camera, backward_point);

    const float diff_col = p.x - backward_point.x;
    const float diff_row = p.y - backward_point.y;
    return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

__device__ void PlaneHypothesisRefinement(const cudaTextureObject_t *images, const cudaTextureObject_t *depth_images, const Camera *cameras, float4 *PlaneHypothesis, float *depth, float *cost, curandState *rand_state, const float *view_weights, const float weight_norm, float4 *PriorPlanes, unsigned int *PlaneMask, float *restricted_cost, const int2 p, const PatchMatchParams params,const int scale)
{
    float perturbation = 0.02f;
    float depth_rand;
    float4 PlaneHypothesisRand;

    const int idx = p.y * cameras[0].width + p.x;
    float gamma = 0.5f;
    float depth_sigma = (params.depth_max - params.depth_min) / 64.0f;
    float two_depth_sigma_squared = 2 * depth_sigma * depth_sigma;
    float angle_sigma = M_PI * (5.0f / 180.0f);
    float two_angle_sigma_squared = 2 * angle_sigma * angle_sigma;
    float beta = 0.18f;
    float depth_prior = 0.0f;

    depth_rand = curand_uniform(rand_state) * (params.depth_max - params.depth_min) + params.depth_min;
    PlaneHypothesisRand = GenerateRandomNormal(cameras[0], p, rand_state);

    float depth_perturbed = *depth;
    const float depth_min_perturbed = (1 - perturbation) * depth_perturbed;
    const float depth_max_perturbed = (1 + perturbation) * depth_perturbed;
    do {//在平面假设附近增加随机扰动
        depth_perturbed = curand_uniform(rand_state) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;
    } while (depth_perturbed < params.depth_min && depth_perturbed > params.depth_max);
    float4 PlaneHypothesisPerturbed = GeneratePerturbedNormal(cameras[0], p, *PlaneHypothesis, rand_state, perturbation * M_PI);

    const int num_planes = 5;
    float depths[num_planes] = {depth_rand, *depth, depth_rand, *depth, depth_perturbed};
    float4 normals[num_planes] = {*PlaneHypothesis, PlaneHypothesisRand, PlaneHypothesisRand, PlaneHypothesisPerturbed, *PlaneHypothesis};

    for (int i = 0; i < num_planes; ++i) {
        float cost_vector[32] = {2.0f};
        float4 temp_plane_hypothesis = normals[i];
        temp_plane_hypothesis.w = GetPlane2Origin(cameras[0], p, depths[i], temp_plane_hypothesis);
        ComputeMultiViewCostVector(images, cameras, p, temp_plane_hypothesis, cost_vector, params,scale);

        float temp_cost = 0.0f;
        for (int j = 0; j < params.num_images - 1; ++j) {
            if (view_weights[j] > 0) {
                if (params.geom_consistency) {
                    temp_cost += view_weights[j] * (cost_vector[j] + 0.2f * ComputeGeomConsistencyCost(depth_images[j], cameras[0], cameras[j+1], temp_plane_hypothesis, p));
                }
                else {
                    temp_cost += view_weights[j] * cost_vector[j];
                }
            }
        }
        temp_cost /= weight_norm;

        float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], temp_plane_hypothesis, p);
        if (params.planar_prior && PlaneMask[idx] > 0) {
            float depth_diff = depths[i] - depth_prior;
            float angle_cos = Vec3DotVec3(PriorPlanes[idx], temp_plane_hypothesis);
            float angle_diff = acos(angle_cos);
            float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
            float restricted_temp_cost = exp(-temp_cost * temp_cost / beta) * prior;
            if (depth_before >= params.depth_min && depth_before <= params.depth_max && restricted_temp_cost > *restricted_cost) {
                *PlaneHypothesis = temp_plane_hypothesis;
                *cost = temp_cost;
            }
        }
        else {
            if (depth_before >= params.depth_min && depth_before <= params.depth_max && temp_cost < *cost) {
                //*depth = depth_before;
                *PlaneHypothesis = temp_plane_hypothesis;
                *cost = temp_cost;
            }
        }

    }
}

__device__ void CheckerboardPropagation(const cudaTextureObject_t *images, const cudaTextureObject_t *depths, const Camera *cameras, float4 *PlaneHypotheses, float *costs, curandState *randStates, unsigned int *selected_views, float4 *PriorPlanes, unsigned int *PlaneMask, const int2 p, const PatchMatchParams params, const int iter,const int scale,float *GeomCost)
{
    int width = cameras[0].width;
    int height = cameras[0].height;
    if (p.x >= width || p.y >= height) {
        return;
    }
    
    const int idx = Point2Idx(p, width);
    curandState* randState = &randStates[idx];
	float4& plane = PlaneHypotheses[idx];

	// static constexpr int2 dirs[8][11] = {
	// 	{{ 0,-1},{-1,-2},{ 1,-2},{-2,-3},{ 2,-3},{-3,-4},{ 3,-4}},
	// 	{{ 0, 1},{-1, 2},{ 1, 2},{-2, 3},{ 2, 3},{-3, 4},{ 3, 4}},
	// 	{{-1, 0},{-2,-1},{-2, 1},{-3,-2},{-3, 2},{-4,-3},{-4, 3}},
	// 	{{ 1, 0},{ 2,-1},{ 2, 1},{ 3,-2},{ 3, 2},{ 4,-3},{ 4, 3}},
	// 	{{0,-3},{0,-5},{0,-7},{0,-9},{0,-11},{0,-13},{0,-15},{0,-17},{0,-19},{0,-21},{0,-1}},
	// 	{{0, 3},{0, 5},{0, 7},{0, 9},{0, 11},{0, 13},{0, 15},{0, 17},{0, 19},{0, 21},{0, 1}},
	// 	{{-3,0},{-5,0},{-7,0},{-9,0},{-11,0},{-13,0},{-15,0},{-17,0},{-19,0},{-21,0},{-1,0}},
	// 	{{ 3,0},{ 5,0},{ 7,0},{ 9,0},{ 11,0},{ 13,0},{ 15,0},{ 17,0},{ 19,0},{ 21,0},{ 1,0}}
	// };
    // static constexpr int numDirs[8] = {7, 7, 7, 7, 11, 11, 11, 11};
    // static constexpr int2 dirs[8][13] = {
    // {{ 6, 5},{ 7, 6},{ 8, 7},{ 9, 8},{ 10, 9},{ 11, 10},{ 12, 11},{ 13, 12},{ 14, 13},{ 15, 14},{ 16, 15},{ 17, 16}},
    // {{-6, 5},{-7, 6},{-8, 7},{-9, 8},{-10, 9},{-11, 10},{-12, 11},{-13, 12},{-14, 13},{-15, 14},{-16, 15},{-17, 16}},
    // {{ 6,-5},{ 7,-6},{ 8,-7},{ 9,-8},{ 10,-9},{ 11,-10},{ 12,-11},{ 13,-12},{ 14,-13},{ 15,-14},{ 16,-15},{ 17,-16}},
    // {{-6,-5},{-7,-6},{-8,-7},{-9,-8},{-10,-9},{-11,-10},{-12,-11},{-13,-12},{-14,-13},{-15,-14},{-16,-15},{-17,-16}},
    // {{0,-5},{0,-7},{0,-9},{0,-11},{0,-13},{0,-15},{0,-17},{0,-19},{0,-21},{0,-23}},
    // {{0, 5},{0, 7},{0, 9},{0, 11},{0, 13},{0, 15},{0, 17},{0, 19},{0, 21},{0, 23}},
    // {{-5,0},{-7,0},{-9,0},{-11,0},{-13,0},{-15,0},{-17,0},{-19,0},{-21,0},{-23,0}},
    // {{ 5,0},{ 7,0},{ 9,0},{ 11,0},{ 13,0},{ 15,0},{ 17,0},{ 19,0},{ 21,0},{ 23,0}}
	// };
	// static constexpr int numDirs[8] = {12, 12, 12, 12, 10, 10, 10, 10};
    // static constexpr int2 dirs[8][12] = {
    // {{-6,-5},{-7,-6},{-8,-7},{-9,-8},{-10,-9},{-11,-10},{-12,-11},{-13,-12},{-14,-13},{-15,-14},{-16,-15},{-17,-16}},
    // {{-6, 5},{-7, 6},{-8, 7},{-9, 8},{-10, 9},{-11, 10},{-12, 11},{-13, 12},{-14, 13},{-15, 14},{-16, 15},{-17, 16}},
    // {{ 6,-5},{ 7,-6},{ 8,-7},{ 9,-8},{ 10,-9},{ 11,-10},{ 12,-11},{ 13,-12},{ 14,-13},{ 15,-14},{ 16,-15},{ 17,-16}},
    // {{ 6, 5},{ 7, 6},{ 8, 7},{ 9, 8},{ 10, 9},{ 11, 10},{ 12, 11},{ 13, 12},{ 14, 13},{ 15, 14},{ 16, 15},{ 17, 16}},
    // {{0,-5},{0,-7},{0,-9},{0,-11},{0,-13},{0,-15},{0,-17},{0,-19},{0,-21},{0,-23}},
    // {{0, 5},{0, 7},{0, 9},{0, 11},{0, 13},{0, 15},{0, 17},{0, 19},{0, 21},{0, 23}},
    // {{-5,0},{-7,0},{-9,0},{-11,0},{-13,0},{-15,0},{-17,0},{-19,0},{-21,0},{-23,0}},
    // {{ 5,0},{ 7,0},{ 9,0},{ 11,0},{ 13,0},{ 15,0},{ 17,0},{ 19,0},{ 21,0},{ 23,0}}
	// };
	// static constexpr int numDirs[8] = {12, 12, 12, 12, 10, 10, 10, 10};
    static constexpr int2 dirs[8][12] = {
    {{-5,-6},{ 5,-6},{-6,-7},{ 6,-7},{-7,-8},{ 7,-8},{-8,-9},{ 8,-9},{-9,-10},{ 9,-10},{-10,-11},{ 10,-11}},
    {{-5, 6},{ 5, 6},{-6, 7},{ 6, 7},{-7, 8},{ 7, 8},{-8, 9},{ 8, 9},{-9, 10},{ 9, 10},{-10, 11},{ 10, 11}},
    {{-6,-5},{-6, 5},{-7,-6},{-7, 6},{-8,-7},{-8, 7},{-9,-8},{-9, 8},{-10,-9},{-10, 9},{-11,-10},{-11, 10}},
    {{ 6,-5},{ 6, 5},{ 7,-6},{ 7, 6},{ 8,-7},{ 8, 7},{ 9,-8},{ 9, 8},{ 10,-9},{ 10, 9},{ 11,-10},{ 11, 10}},
    {{0,-5},{0,-7},{0,-9},{0,-11},{0,-13},{0,-15},{0,-17},{0,-19},{0,-21},{0,-23}},
    {{0, 5},{0, 7},{0, 9},{0, 11},{0, 13},{0, 15},{0, 17},{0, 19},{0, 21},{0, 23}},
    {{-5,0},{-7,0},{-9,0},{-11,0},{-13,0},{-15,0},{-17,0},{-19,0},{-21,0},{-23,0}},
    {{ 5,0},{ 7,0},{ 9,0},{ 11,0},{ 13,0},{ 15,0},{ 17,0},{ 19,0},{ 21,0},{ 23,0}}
	};
	static constexpr int numDirs[8] = {12, 12, 12, 12, 10, 10, 10, 10};
	const int neighborPositions[4] = {
		idx - width,
		idx + width,
		idx - 1,
		idx + 1,
	};
    int positions[8];
    float cost_array[8][32] = {2.0f};
    bool flag[8] = {false};

	for (int posId=0; posId<8; ++posId) {
		const int2* samples = dirs[posId];
		int2 bestNx; 
        float bestConf(FLT_MAX);
		for (int dirId=0; dirId<numDirs[posId]; ++dirId) {
			const int2& offset = samples[dirId];
			const int2 np=make_int2(p.x+offset.x, p.y+offset.y);
			if (!(np.x>=0 && np.y>=0 && np.x<width && np.y<height))
				continue;
			const int nidx = Point2Idx(np, width);
			const float nconf = costs[nidx];
			if (bestConf > nconf) {
				bestNx = np;
				bestConf = nconf;
			}
		}
		if (bestConf < FLT_MAX) {
			flag[posId]=true;
			positions[posId] = Point2Idx(bestNx, width);
			ComputeMultiViewCostVector(images, cameras, p, PlaneHypotheses[positions[posId]], cost_array[posId], params,scale);
		}
	}

    float view_weights[32] = {0.0f};
    float view_selection_priors[32] = {0.0f};
    for (int i = 0; i < 4; ++i) {
        if (flag[i]) {
            for (int j = 0; j < params.num_images - 1; ++j) {
                view_selection_priors[j] += (isSet(selected_views[neighborPositions[i]], j) ? 0.9f : 0.1f);
            }
        }
    }

    float sampling_probs[32] = {0.0f};
    float cost_threshold = 0.8 * expf((iter) * (iter) / (-90.0f));
    for (int i = 0; i < params.num_images - 1; i++) {
        float count = 0;
        int count_false = 0;
        float tmpw = 0;
        for (int j = 0; j < 8; j++) {
            if (cost_array[j][i] < cost_threshold) {
                tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-0.18f));
                count++;
            }
            if (cost_array[j][i] > 1.2f) {
                count_false++;
            }
        }
        if (count > 2 && count_false < 3) {
            sampling_probs[i] = view_selection_priors[i]*tmpw / count;
        }
        else if (count_false < 3) {
            sampling_probs[i] = view_selection_priors[i]*expf(cost_threshold * cost_threshold / (-0.32f));
        }else{
            sampling_probs[i] = 0;
        }
    }
    TransformPDFToCDF(sampling_probs, params.num_images - 1);
    for (int sample = 0; sample < 15; ++sample) {
        const float rand_prob = curand_uniform(&randStates[idx]) - FLT_EPSILON;

        for (int image_id = 0; image_id < params.num_images - 1; ++image_id) {
            const float prob = sampling_probs[image_id];
            if (prob > rand_prob) {
                view_weights[image_id] += 1.0f;
                break;
            }
        }
    }

    unsigned int temp_selected_views = 0;
    int num_selected_view = 0;
    float weight_norm = 0;
    for (int i = 0; i < params.num_images - 1; ++i) {
        if (view_weights[i] > 0) {
            setBit(temp_selected_views, i);
            weight_norm += view_weights[i];
            num_selected_view++;
        }
    }

    float final_costs[8] = {0.0f};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < params.num_images - 1; ++j) {
            if (view_weights[j] > 0) {
                if (params.geom_consistency) {
                    if (flag[i]) {
                        final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.2f * ComputeGeomConsistencyCost(depths[j], cameras[0], cameras[j+1], PlaneHypotheses[positions[i]], p));
                    }
                    else {
                        final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.1f * 3.0f);
                    }
                }
                else {
                    final_costs[i] += view_weights[j] * cost_array[i][j];
                }
            }
        }
        final_costs[i] /= weight_norm;
    }

    const int min_cost_idx = FindMinCostIndex(final_costs, 8);

    float cost_vector_now[32] = {2.0f};
    ComputeMultiViewCostVector(images, cameras, p, PlaneHypotheses[idx], cost_vector_now, params,scale);
    float cost_now = 0.0f;
    for (int i = 0; i < params.num_images - 1; ++i) {
        if (params.geom_consistency) {
            float GeomCostTemp = 0.2f * ComputeGeomConsistencyCost(depths[i], cameras[0], cameras[i+1], PlaneHypotheses[idx], p);
            cost_now += view_weights[i] * (cost_vector_now[i] + GeomCostTemp);
            GeomCost[idx] += view_weights[i] * GeomCostTemp;
        }
        else {
            cost_now += view_weights[i] * cost_vector_now[i];
        }
    }
    cost_now /= weight_norm;
    if (params.geom_consistency){
        GeomCost[idx] /= weight_norm;
    }
    costs[idx] = cost_now;
    float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], PlaneHypotheses[idx], p);
    float restricted_cost = 0.0f;
    if (params.planar_prior&&!params.geom_consistency) {
        float restricted_final_costs[8] = {0.0f};
        float gamma = 0.5f;
        float depth_sigma = (params.depth_max - params.depth_min) / 64.0f;
        float two_depth_sigma_squared = 2 * depth_sigma * depth_sigma;
        float angle_sigma = M_PI * (5.0f / 180.0f);
        float two_angle_sigma_squared = 2 * angle_sigma * angle_sigma;
        float depth_prior = ComputeDepthfromPlaneHypothesis(cameras[0], PriorPlanes[idx], p);
        float beta = 0.18f;

        if (PlaneMask[idx] > 0) {
            for (int i = 0; i < 8; i++) {
                if (flag[i]) {
                    float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], PlaneHypotheses[positions[i]], p);
                    float depth_diff = depth_now - depth_prior;
                    float angle_cos = Vec3DotVec3(PriorPlanes[idx], PlaneHypotheses[positions[i]]);
                    float angle_diff = acos(angle_cos);
                    float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
                    restricted_final_costs[i] = exp(-final_costs[i] * final_costs[i] / beta) * prior;
                }
            }
            const int max_cost_idx = FindMaxCostIndex(restricted_final_costs, 8);

            float restricted_cost_now = 0.0f;
            float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], PlaneHypotheses[idx], p);
            float depth_diff = depth_now - depth_prior;
            float angle_cos = Vec3DotVec3(PriorPlanes[idx], PlaneHypotheses[idx]);
            float angle_diff = acos(angle_cos);
            float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
            restricted_cost_now = exp(-cost_now * cost_now / beta) * prior;

            if (flag[max_cost_idx]) {
                float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], PlaneHypotheses[positions[max_cost_idx]], p);

                if (depth_before >= params.depth_min && depth_before <= params.depth_max && restricted_final_costs[max_cost_idx] > restricted_cost_now) {
                    depth_now   = depth_before;
                    PlaneHypotheses[idx] = PlaneHypotheses[positions[max_cost_idx]];
                    costs[idx] = final_costs[max_cost_idx];
                    restricted_cost = restricted_final_costs[max_cost_idx];
                    selected_views[idx] = temp_selected_views;
                }
            }
        }
        else if (flag[min_cost_idx]) {
            float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], PlaneHypotheses[positions[min_cost_idx]], p);

            if (depth_before >= params.depth_min && depth_before <= params.depth_max && final_costs[min_cost_idx] < cost_now) {
                depth_now = depth_before;
                PlaneHypotheses[idx] = PlaneHypotheses[positions[min_cost_idx]];
                costs[idx] = final_costs[min_cost_idx];
            }
        }
    }


    float4 plane_hypotheses_now=PlaneHypotheses[idx];
    if (!params.planar_prior &&flag[min_cost_idx]) {
        float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], PlaneHypotheses[positions[min_cost_idx]], p);

        if (depth_before >= params.depth_min && depth_before <= params.depth_max && final_costs[min_cost_idx] < cost_now) {
           depth_now = depth_before;
           plane_hypotheses_now = PlaneHypotheses[positions[min_cost_idx]];
           cost_now = final_costs[min_cost_idx];
           selected_views[idx] = temp_selected_views;
        }
    }

    PlaneHypothesisRefinement(images, depths, cameras, &plane_hypotheses_now, &depth_now, &cost_now, &randStates[idx], view_weights, weight_norm, PriorPlanes, PlaneMask, &restricted_cost, p, params,scale);
    costs[idx] = cost_now;
    PlaneHypotheses[idx] = plane_hypotheses_now;

}

__global__ void BlackPixelUpdate(const cudaTextureObject_t* images, const cudaTextureObject_t *depths, Camera *cameras, float4 *PlaneHypotheses, float *costs, curandState *randStates, unsigned int *selected_views, float4 *PriorPlanes, unsigned int *PlaneMask, const PatchMatchParams params, const int iter,const int scale,float *GeomCost){
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2;
    } else {
        p.y = p.y * 2 + 1;
    }
    CheckerboardPropagation(images, depths, cameras, PlaneHypotheses, costs, randStates, selected_views, PriorPlanes, PlaneMask, p, params, iter,scale,GeomCost);

}

__global__ void RedPixelUpdate(const cudaTextureObject_t* images, const cudaTextureObject_t *depths, Camera *cameras, float4 *PlaneHypotheses, float *costs, curandState *randStates, unsigned int *selected_views, float4 *PriorPlanes, unsigned int *PlaneMask, const PatchMatchParams params, const int iter,const int scale,float *GeomCost){
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2 + 1;
    } else {
        p.y = p.y * 2;
    }
    CheckerboardPropagation(images, depths, cameras, PlaneHypotheses, costs, randStates, selected_views, PriorPlanes, PlaneMask, p, params, iter,scale,GeomCost);
}

__global__ void GetDepthandNormal(Camera *cameras, float4 *plane_hypotheses, const PatchMatchParams params)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = cameras[0].width;
    const int height = cameras[0].height;

    if (p.x >= width || p.y >= height) {
        return;
    }

    const int idx = p.y * width + p.x;
    plane_hypotheses[idx].w = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[idx], p);
    plane_hypotheses[idx] = TransformNormal(cameras[0], plane_hypotheses[idx]);
}

__device__ void CheckerboardFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs, const int2 p)
{
    int width = cameras[0].width;
    int height = cameras[0].height;
    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;

    float filter[21];
    int index = 0;

    filter[index++] = plane_hypotheses[center].w;

    // Left
    const int left = center - 1;
    const int leftleft = center - 3;

    // Up
    const int up = center - width;
    const int upup = center - 3 * width;

    // Down
    const int down = center + width;
    const int downdown = center + 3 * width;

    // Right
    const int right = center + 1;
    const int rightright = center + 3;

    if (costs[center] < 0.001f) {
        return;
    }

    if (p.y>0) {
        filter[index++] = plane_hypotheses[up].w;
    }
    if (p.y>2) {
        filter[index++] = plane_hypotheses[upup].w;
    }
    if (p.y>4) {
        filter[index++] = plane_hypotheses[upup-width*2].w;
    }
    if (p.y<height-1) {
        filter[index++] = plane_hypotheses[down].w;
    }
    if (p.y<height-3) {
        filter[index++] = plane_hypotheses[downdown].w;
    }
    if (p.y<height-5) {
        filter[index++] = plane_hypotheses[downdown+width*2].w;
    }
    if (p.x>0) {
        filter[index++] = plane_hypotheses[left].w;
    }
    if (p.x>2) {
        filter[index++] = plane_hypotheses[leftleft].w;
    }
    if (p.x>4) {
        filter[index++] = plane_hypotheses[leftleft-2].w;
    }
    if (p.x<width-1) {
        filter[index++] = plane_hypotheses[right].w;
    }
    if (p.x<width-3) {
        filter[index++] = plane_hypotheses[rightright].w;
    }
    if (p.x<width-5) {
        filter[index++] = plane_hypotheses[rightright+2].w;
    }
    if (p.y>0 &&
        p.x<width-2) {
        filter[index++] = plane_hypotheses[up+2].w;
    }
    if (p.y< height-1 &&
        p.x<width-2) {
        filter[index++] = plane_hypotheses[down+2].w;
    }
    if (p.y>0 &&
        p.x>1)
    {
        filter[index++] = plane_hypotheses[up-2].w;
    }
    if (p.y<height-1 &&
        p.x>1) {
        filter[index++] = plane_hypotheses[down-2].w;
    }
    if (p.x>0 &&
        p.y>2)
    {
        filter[index++] = plane_hypotheses[left  - width*2].w;
    }
    if (p.x<width-1 &&
        p.y>2)
    {
        filter[index++] = plane_hypotheses[right - width*2].w;
    }
    if (p.x>0 &&
        p.y<height-2) {
        filter[index++] = plane_hypotheses[left  + width*2].w;
    }
    if (p.x<width-1 &&
        p.y<height-2) {
        filter[index++] = plane_hypotheses[right + width*2].w;
    }

    sort_small(filter,index);
    int median_index = index / 2;
    if (index % 2 == 0) {
        plane_hypotheses[center].w = (filter[median_index-1] + filter[median_index]) / 2;
    } else {
        plane_hypotheses[center].w = filter[median_index];
    }
}

__global__ void BlackPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2;
    } else {
        p.y = p.y * 2 + 1;
    }

    CheckerboardFilter(cameras, plane_hypotheses, costs, p);
}

__global__ void RedPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2 + 1;
    } else {
        p.y = p.y * 2;
    }

    CheckerboardFilter(cameras, plane_hypotheses, costs, p);
}

__device__ void ProjectPoint2(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

void PatchMatchCUDA::Run(){
    const int width=cameras[0].width;
    const int height=cameras[0].height;
    int maxscale=params.max_scale;
    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);
    const dim3 blockSize(BLOCK_W,BLOCK_H,1);
    const dim3 gridSizeInit((width+BLOCK_W-1)/BLOCK_W,(height+BLOCK_H-1)/BLOCK_H,1);
	const dim3 gridSizeCheckerboard((width + BLOCK_W - 1) / BLOCK_W, ((height / 2) + BLOCK_H - 1) / BLOCK_H, 1);

    int max_iterations = params.max_iterations;
    
    //深度图初始化
    InitializeScore<<<gridSizeInit,blockSize>>>(cudaTextureImages,cudaCameras,cudaPlaneHypotheses,cudaCosts,cudaRandStates,cudaSelectedViews,cudaPriorPlanes,cudaPlaneMask,params,maxscale);
    cudaDeviceSynchronize();
    if(params.geom_consistency||params.planar_prior){
        for (int i = 0; i < max_iterations; ++i) {
            BlackPixelUpdate<<<gridSizeCheckerboard, blockSize>>>(cudaTextureImages, cudaTextureDepths, cudaCameras, cudaPlaneHypotheses, cudaCosts, cudaRandStates, cudaSelectedViews, cudaPriorPlanes, cudaPlaneMask, params, i, 0, cudaGeomCosts);
            checkCudaCall(cudaDeviceSynchronize());
            RedPixelUpdate<<<gridSizeCheckerboard, blockSize>>>(cudaTextureImages, cudaTextureDepths, cudaCameras, cudaPlaneHypotheses, cudaCosts, cudaRandStates, cudaSelectedViews, cudaPriorPlanes, cudaPlaneMask, params, i, 0, cudaGeomCosts);
            checkCudaCall(cudaDeviceSynchronize());
            printf("iteration: %d/%d\n", i+1,max_iterations);
        }
    }
    else{
        for(int scale=maxscale;scale>=0;--scale)
        {
            printf("Scale: %d\n", scale);
            for (int i = 0; i < max_iterations; ++i) {
                BlackPixelUpdate<<<gridSizeCheckerboard, blockSize>>>(cudaTextureImages, cudaTextureDepths, cudaCameras, cudaPlaneHypotheses, cudaCosts, cudaRandStates, cudaSelectedViews, cudaPriorPlanes, cudaPlaneMask, params, i, scale, cudaGeomCosts);
                checkCudaCall(cudaDeviceSynchronize());
                RedPixelUpdate<<<gridSizeCheckerboard, blockSize>>>(cudaTextureImages, cudaTextureDepths, cudaCameras, cudaPlaneHypotheses, cudaCosts, cudaRandStates, cudaSelectedViews, cudaPriorPlanes, cudaPlaneMask, params, i, scale, cudaGeomCosts);
                checkCudaCall(cudaDeviceSynchronize());
                printf("iteration: %d/%d\n", i+1,max_iterations);
            }
        }
    }

    GetDepthandNormal<<<gridSizeInit,blockSize>>>(cudaCameras, cudaPlaneHypotheses, params);
    checkCudaCall(cudaDeviceSynchronize());
    
    BlackPixelFilter<<<gridSizeCheckerboard, blockSize>>>(cudaCameras, cudaPlaneHypotheses, cudaCosts);
    checkCudaCall(cudaDeviceSynchronize());
    RedPixelFilter<<<gridSizeCheckerboard, blockSize>>>(cudaCameras, cudaPlaneHypotheses, cudaCosts);
    checkCudaCall(cudaDeviceSynchronize());

    checkCudaCall(cudaMemcpy(hostPlaneHypotheses, cudaPlaneHypotheses, sizeof(float4) * width * height, cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(hostCosts, cudaCosts, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
    if(params.geomPlanarPrior)
    {
        checkCudaCall(cudaMemcpy(hostGeomCosts, cudaGeomCosts, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
        // TextureConfMap<<<gridSizeInit,blockSize>>>(cudaTextureImages,cudaTexCofMap,width,height,params);
        // checkCudaCall(cudaDeviceSynchronize());
        // checkCudaCall(cudaMemcpy(hostTexCofMap, cudaTexCofMap, sizeof(uchar)*width*height, cudaMemcpyDeviceToHost));
    }

        

}
