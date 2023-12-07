#include "PatchMatch.h"

#ifdef BUILD_NCNN
void GenerateSkyRegionMask(std::vector<Scene> &Scenes, std::string &project_path, std::string &dense_folder, const int max_image_size){
    std::string param_path = project_path + "/segment_model/skysegsmall_sim-opt-fp16.param";
    std::string model_path = project_path + "/segment_model/skysegsmall_sim-opt-fp16.bin";
    SkySegment Skyseg(param_path.data(),model_path.data());

    std::string image_folder = dense_folder + std::string("/images");
    int n=Scenes.size();
    for(int i=0;i<n;++i){
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << Scenes[i].refID << ".jpg";
        cv::Mat bgr = cv::imread (image_path.str(), 1);
        cv::Mat dst = bgr.clone();
        while (dst.rows > 768 && dst.cols >768 ) {
            pyrDown(dst, dst, cv::Size(dst.cols / 2, dst.rows / 2));
        }
        cv::Mat opencv_mask = Skyseg.maskExtractor(dst);

        //
        int w = bgr.cols;
        int h = bgr.rows;
        if(bgr.cols>max_image_size||bgr.rows>max_image_size)
        {
            const float factor_x = static_cast<float>(max_image_size) / bgr.cols;
            const float factor_y = static_cast<float>(max_image_size) / bgr.rows;
            const float factor = std::min(factor_x, factor_y);

            w = std::round(bgr.cols * factor);
            h = std::round(bgr.rows * factor);
        }
        //
        cv::resize(opencv_mask,opencv_mask,cv::Size(w,h),cv::INTER_LINEAR);

        std::stringstream result_path;
        result_path << dense_folder << "/MPMVS" << "/2333_" << std::setw(8) << std::setfill('0') << Scenes[i].refID;
        std::string result_folder = result_path.str();
        //std::cout<<result_folder<<std::endl;
        mkdir(result_folder.c_str(), 0777);
        std::string mask_path = result_folder + "/skymask.jpg";
        std::string refinemask_path = result_folder + "/skymask_refine.jpg";
        cv::imwrite(mask_path, 255*opencv_mask);

        mask_refine(image_path.str(),mask_path,refinemask_path);
    }

}
#endif

void checkCudaCall(const cudaError_t error) {
	if (error == cudaSuccess)
		return;
    std::cout<<cudaGetErrorString(error)<<"heppend!"<< std::endl;
	exit(EXIT_FAILURE);
}

void GenerateSampleList(const ConfigParams &config, std::vector<Scene> &Scenes){
    Scenes.clear();
    std::string cluster_list_path = config.input_folder + std::string("/pair.txt");
    std::ifstream file(cluster_list_path);
    if(!file.is_open()){
        std::cout<<"can not open file in path:   "<<cluster_list_path<<std::endl;
        exit(1);
    }

    int num_images;
    file >> num_images;
    const int maxSourceImageNum = config.MaxSourceImageNum;
    const int maxImageSize = config.MaxImageSize;
    for (int i = 0; i < num_images; ++i) {
        Scene scene;
        scene.max_image_size = maxImageSize;
        scene.srcID.clear();
        file >> scene.refID;
        scene.srcID.push_back(scene.refID);

        int num_src_images;
        file >> num_src_images;
        for (int j = 0; j < num_src_images; ++j) {
            int id;
            float score;
            file >> id >> score;
            if (score <= 0.0f) {
                continue;
            }
            if(j < maxSourceImageNum)
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
    outputPly = fopen(plyFilePath.c_str(), "wb");

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
//#pragma omp parallel for
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
//#pragma omp critical
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

    pointX.x = tmpX.x + camera.C[0];
    pointX.y = tmpX.y + camera.C[1];
    pointX.z = tmpX.z + camera.C[2];

    return pointX;
}

cv::Vec3f TransformNormalonWorld(const Camera camera, cv::Vec3f normal)
{
    cv::Vec3f transformed_normal;
    transformed_normal[0] = camera.R[0] * normal[0] + camera.R[3] * normal[1] + camera.R[6] * normal[2];
    transformed_normal[1] = camera.R[1] * normal[0] + camera.R[4] * normal[1] + camera.R[7] * normal[2];
    transformed_normal[2] = camera.R[2] * normal[0] + camera.R[5] * normal[1] + camera.R[8] * normal[2];
    return transformed_normal;
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

void RunFusion(const std::string &dense_folder, const std::string &out_folder, const std::vector<Scene> &Scenes, bool sky_mask, bool use_prior_map){
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

    std::vector<cv::Mat> prior_depths;
    std::vector<cv::Mat_<cv::Vec3f>> prior_normals;
    prior_depths.clear();
    prior_normals.clear();
    
    std::map<int, int> image_id_2_index;

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
        cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);

        RescaleImageAndCamera(image, scaled_image, depth, camera);
        images.push_back(scaled_image);
        cameras.push_back(camera);
        depths.push_back(depth);
        normals.push_back(normal);
        masks.push_back(mask);
    }

    std::vector<PointList> PointCloud;
    PointCloud.clear();

    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        cv::Mat skymask;
         if(sky_mask){
             std::stringstream result_path;
             result_path << dense_folder << "/MPMVS" << "/2333_" << std::setw(8) << std::setfill('0') << Scenes[i].refID;
             std::string mask_path =result_path.str()+ "/skymask_refine.jpg";
             cv::Mat temp=cv::imread(mask_path,1);
            
             const int cols = skymask.cols;
             const int rows = skymask.rows;

             if (cols != depths[i].cols || rows != depths[i].rows) {
                 cv::resize(temp, skymask, cv::Size(depths[i].cols,depths[i].rows), 0, 0, cv::INTER_LINEAR);
             }
            
         }

        const int cols = depths[i].cols;
        const int rows = depths[i].rows;
        int num_ngb = Scenes[i].srcID.size();
        std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
        for (int r =0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if(sky_mask && skymask.at<float>(r,c)!=0){
                     masks[i].at<uchar>(r, c) = 1;
                     continue;
                }

                if (masks[i].at<uchar>(r, c) == 1)
                    continue;

                float ref_depth = depths[i].at<float>(r, c);
                if (ref_depth <= 0.0 )
                    continue;

                cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);
                float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
                float3 consistent_Point = PointX;
                cv::Vec3f consistent_normal = ref_normal;
                float consistent_Color[3] = {(float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2]};
                int num_consistent = 0;
                float dynamic_consistency = 0;

                for (int j = 1; j < num_ngb; ++j) {
                    if(j == num_ngb - 1 && num_consistent == 0)
                        break;

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
                        if (src_depth <= 0.0)
                            continue;

                        cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
                        float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
                        float2 tmp_pt;
                        ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
                        float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                        if (reproj_error > 2.0f)
                            continue;

                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        if(relative_depth_diff > 0.01f)
                            continue;

                        float angle = GetAngle(ref_normal, src_normal);
                        if (angle < 0.174533f) {
                            used_list[j].x = src_c;
                            used_list[j].y = src_r;

                            float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
                            float cons = exp(-tmp_index);
                            dynamic_consistency += exp(-tmp_index);
                            num_consistent++;
//                            used_list[j].x = src_c;
//                            used_list[j].y = src_r;
//                            num_consistent++;
                        }
                    }
                }
                if (num_consistent >= 2 ) {//num_consistent >= 1 && (dynamic_consistency > 0.3 * num_consistent)//num_consistent >= 2
                    PointList point3D;
                    point3D.coord = consistent_Point;
                    point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                    point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                    PointCloud.push_back(point3D);

                    for (int j = 1; j < num_ngb; ++j) {
                        if (used_list[j].x == -1)
                            continue;
                        masks[image_id_2_index[Scenes[i].srcID[j]]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
                    }
                }
            }
            
        }
        // std::stringstream ply_path ;
        // ply_path  << out_folder<< "/" << std::setw(8) << std::setfill('0') << Scenes[i].refID << ".ply";
        // StoreColorPlyFileBinaryPointCloud (ply_path.str(), PointCloud);
        // PointCloud.clear();
    }
    std::string ply_path = out_folder + "/MPMVS_model.ply";
    StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);
}

void ProcessProblem(const std::string &input_folder,const std::string &output_folder, std::vector<Scene> &Scenes, const int ID,bool geom_consistency=false, bool planar_prior=false){
    Scene& scene = Scenes[ID];
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << scene.refID << " ..." << std::endl;
    cudaSetDevice(0);
    std::stringstream result_path;
    result_path << output_folder << "/2333_" << std::setw(8) << std::setfill('0') << scene.refID;
    std::string result_folder = result_path.str();
    mkdir(result_folder.c_str(), 0777);

    //Run PatchMatch
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
    cv::Mat_<uchar> TexCofMap = cv::Mat::zeros(height, width, CV_32FC1);

    if (planar_prior) {
        std::cout << "Run Planar Prior PatchMatch MVS ";
        MP.SetPlanarPriorParams();
        MP.SetGeomConsistencyParams(false,true);
        const cv::Rect imageRC(0, 0, width, height);

        std::vector<cv::Point> Vertices;
        MP.GetTriangulateVertices(Vertices);
        const auto triangles = MP.DelaunayTriangulation(imageRC, Vertices);

        cv::Mat_<float> mask_tri = cv::Mat::zeros(height, width, CV_32FC1);
        std::vector<float4> planeParams_tri;
        planeParams_tri.clear();

        cv::Mat refImage = MP.GetReferenceImage().clone();
        std::vector<cv::Mat> mbgr(3);
        mbgr[0] = refImage.clone();
        mbgr[1] = refImage.clone();
        mbgr[2] = refImage.clone();
        cv::Mat image_tri;
        cv::merge(mbgr, image_tri);
        // for (const auto triangle : triangles) {
        //     if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
        //         cv::line(image_tri, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
        //         cv::line(image_tri, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
        //         cv::line(image_tri, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
        //     }
        // }

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
                ++idx;

                //caculate area
                float S = 1.0f;
                {
                    float x01 = triangle.pt1.x - triangle.pt2.x;
                    float y01 = triangle.pt1.y - triangle.pt2.y;
                    float x02 = triangle.pt1.x - triangle.pt3.x;
                    float y02 = triangle.pt1.y - triangle.pt3.y;
                    S = fabs(x01 * y02 - x02 * y01) * 0.5;
                }
                //caculate angle diff
                float aver_angle = 0.0f;
                {
                    // int idx = triangle.pt1.y * width + triangle.pt1.y;
                    // float3 ni = MP.GetNormalonRef(idx);
                    // float dot_product = ni.x * n4.x + ni.y * n4.y + ni.z * n4.z;
                    aver_angle += MP.GetAngleDiff(triangle.pt1, n4, width);
                    aver_angle += MP.GetAngleDiff(triangle.pt2, n4, width);
                    aver_angle += MP.GetAngleDiff(triangle.pt3, n4, width);
                }
                aver_angle /= 3;
                //
                //if (aver_angle < 5.f) 
                {
                cv::line(image_tri, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
                cv::line(image_tri, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
                cv::line(image_tri, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
                }
            }
        }
//        cv::Mat_<cv::Vec3f> priornormals = cv::Mat::zeros(height, width, CV_32FC3);
//        cv::Mat_<float> priordepths = cv::Mat::zeros(height, width, CV_32FC1);
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                if (mask_tri(j, i) > 0) {
                    const float4 n4 = planeParams_tri[mask_tri(j, i) - 1];
                    float d = MP.GetDepthFromPlaneParam(n4, i, j);
                    if (d <= MP.GetMaxDepth() && d >= MP.GetMinDepth()) {
                        //priordepths(j, i) = d;
                    }else{
                        mask_tri(j, i) = 0;
                    }
                }
            }
        }

        std::string triangulation_path = result_folder + "/triangulation.png";
        cv::imwrite(triangulation_path, image_tri);

//        std::string depth_path = result_folder + "/depths_prior.dmb";
//        std::string normal_path = result_folder + "/normal_prior.dmb";
//        writeDepthDmb(depth_path, priordepths);
//        writeNormalDmb(normal_path, priornormals);
        MP.CudaPlanarPriorInitialization(planeParams_tri, mask_tri);
        std::cout << " ..." << std::endl;
        MP.Run();
        MP.SetGeomConsistencyParams(geom_consistency, planar_prior);

    }
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col){
            int idx = row * width + col;
            float4 plane_hypothesis = MP.GetPlaneHypothesis(idx);
            depths(row, col) = plane_hypothesis.w;
            normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            costs(row, col) = MP.GetCost(idx);
//            if(geom_consistency && planar_prior)
//                TexCofMap(row, col) = MP.GetTextureCofidence(idx);
        }
    }

    std::string suffix = "/depths.dmb";
    std::string depth_path = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path = result_folder + "/costs.dmb";
    std::string cost_path2 = result_folder + "/costs.jpg";
    std::string texcof_path = result_folder + "/TexCofMap.jpg";
    
    cv::Mat uint_dmb;
    costs.convertTo(uint_dmb,CV_8U,255.0/2.0);
    cv::imwrite(cost_path2,uint_dmb);


    writeDepthDmb(depth_path, depths);
    writeNormalDmb(normal_path, normals);
    writeDepthDmb(cost_path, costs);
//    if(geom_consistency && planar_prior)
//        cv::imwrite(texcof_path,TexCofMap);
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << scene.refID << " done!" << std::endl;

    MP.Release(Scenes, ID);
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
    if(geom_consistency){
        params.max_iterations = 2;
        params.geomPlanarPrior = planar_prior;
    }

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

float PatchMatchCUDA::GetAngleDiff(const cv::Point tri, const float4 n4, const int width)
{
    const int index = tri.y * width + tri.y;
    float4 PointX = hostPlaneHypotheses[index];
    const Camera &camera = cameras[0];
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    float dot_product = tmp.x * n4.x + tmp.y * tmp.y + tmp.z * n4.z;
    return acosf(dot_product);
}

float PatchMatchCUDA::GetCost(const int index)
{
    return hostCosts[index];
}

uchar PatchMatchCUDA::GetTextureCofidence(const int index){
    return hostTexCofMap[index];
}

float PatchMatchCUDA::GetGeomCost(const int index){
    return hostGeomCosts[index];
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

float4 PatchMatchCUDA::GetPriorPlaneParams(const Triangle triangle, int width)
{
    cv::Mat A(3, 4, CV_32FC1);
    cv::Mat B(4, 1, CV_32FC1);
    float3 ptX1 = Get3DPointonRefCam(triangle.pt1.x, triangle.pt1.y, GetPlaneHypothesis(triangle.pt1.y * width + triangle.pt1.x).w, cameras[0]);
    float3 ptX2 = Get3DPointonRefCam(triangle.pt2.x, triangle.pt2.y, GetPlaneHypothesis(triangle.pt2.y * width + triangle.pt2.x).w, cameras[0]);
    float3 ptX3 = Get3DPointonRefCam(triangle.pt3.x, triangle.pt3.y, GetPlaneHypothesis(triangle.pt3.y * width + triangle.pt3.x).w, cameras[0]);    

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

    cv::Subdiv2D subdiv2d(boundRC);
    for (const auto point : points) {
        subdiv2d.insert(cv::Point2f((float)point.x, (float)point.y));
    }

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
    if(!params.geomPlanarPrior){
        for (int row = 0; row < height; row += step_size) {
            for (int col = 0; col < width; col += step_size) {
                float min_cost = 2.0f;
                cv::Point temp_point;
                int c_bound = std::min(width, col + step_size);
                int r_bound = std::min(height, row + step_size);
                for (int r = row; r < r_bound; ++r) {
                    int idx = r * width + col;
                    for (int c = col; c < c_bound; ++c) {
                        float cost = GetCost(idx);
                        if (cost < 2.0f && min_cost > cost) {
                            temp_point = cv::Point(c, r);
                            min_cost = cost;
                        }
                        ++idx;
                    }
                }
                if(min_cost < 0.1f)
                    Vertices.push_back(temp_point);
            }
        }
    }else{
//        for (int row = 0; row < height; row += step_size) {
//            for (int col = 0; col < width; col += step_size) {
//                float minCosts[3] = {2.0f,2.0f,2.0f};
//                std::vector<cv::Point> points(3);
//                cv::Point temp_point;
//                int c_bound = std::min(width, col + step_size);
//                int r_bound = std::min(height, row + step_size);
//                for (int r = row; r < r_bound; ++r) {
//                    int idx = r * width + col;
//                    for (int c = col; c < c_bound; ++c) {
//                        float cost = GetCost(idx);
//                        if (cost < 0.16f && GetGeomCost(idx)< 0.3f && cost < minCosts[2]){
//                            minCosts[2] = cost;
//                            points[2] = cv::Point(c, r);
//                            for(int i = 1; i >= 0; --i){
//                                if(minCosts[i] <= minCosts[i+1]) break;
//
//                                float temp = minCosts[i+1];
//                                minCosts[i+1] = minCosts[i];
//                                minCosts[i] = temp;
//
//                                cv::Point tp = points[i+1];
//                                points[i+1] = points[i];
//                                points[i] = tp;
//                            }
//                        }
//                        ++idx;
//                    }
//                }
//                for(int i = 0; i <3; ++i){
//                    if(minCosts[i] < 0.2f){
//                        Vertices.push_back(points[i]);
//                    }else
//                        break;
//                }
//            }
//        }
        for (int row = 0; row < height; row += step_size) {
            for (int col = 0; col < width; col += step_size) {
                float minCosts[3] = {2.0f,2.0f,2.0f};
                std::vector<cv::Point> points(3);
                cv::Point temp_point;
                int c_bound = std::min(width, col + step_size);
                int r_bound = std::min(height, row + step_size);
                float cost_sum = 0.0f;
                for (int r = row; r < r_bound; ++r) {
                    int idx = r * width + col;
                    for (int c = col; c < c_bound; ++c) {
                        const float cost = GetCost(idx);
                        cost_sum += cost;
                        if (cost < 0.5f && GetGeomCost(idx) < 0.3f && cost < minCosts[2]){//cost < 0.16f && GetGeomCost(idx)< 0.3f && cost < minCosts[2]
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
                        }
                        ++idx;
                    }
                }
                cost_sum = cost_sum / (r_bound * c_bound) * 0.75;
                const float thresh_cost = std::max(cost_sum , 0.2f);
                for(int i = 0; i <3; ++i){
                    if(minCosts[i] < thresh_cost){
                        Vertices.push_back(points[i]);
                    }else
                        break;
                }
            }
        }
        // for (int row = 0; row < height; row += step_size) {
        //     for (int col = 0; col < width; col += step_size) {
        //         float minCosts[5] = {2.0f,2.0f,2.0f,2.0f,2.0f};
        //         std::vector<cv::Point> points(5);
        //         cv::Point temp_point;
        //         int c_bound = std::min(width, col + step_size);
        //         int r_bound = std::min(height, row + step_size);
        //         int valid_num = 0;
        //         for (int r = row; r < r_bound; ++r) {
        //             int idx = r * width + col;
        //             for (int c = col; c < c_bound; ++c) {
        //                 float cost = GetCost(idx);
        //                 if (cost < 0.16f && GetGeomCost(idx)< 0.3f && cost < minCosts[4]){
        //                     ++valid_num;
        //                     minCosts[4] = cost;
        //                     points[4] = cv::Point(c, r);
        //                     for(int i = 3; i >= 0; --i){
        //                         if(minCosts[i] <= minCosts[i+1]) break;
                                
        //                         float temp = minCosts[i+1];  
        //                         minCosts[i+1] = minCosts[i];
        //                         minCosts[i] = temp;
                                
        //                         cv::Point tp = points[i+1];
        //                         points[i+1] = points[i];
        //                         points[i] = tp;
        //                     }
        //                 }
        //                 ++idx;
        //             }
        //         }
        //         if(valid_num < 3) continue;
        //         int x0 = -1, y0 = -1;
        //         int num = 0;
        //         for(int i = 0; i <5; ++i){
        //             const cv::Point &p = points[i];
        //             if(minCosts[i] < 0.2f && num <3){
        //                 if((abs(x0 - p.x) + abs(y0 - p.y)) <2) continue;
        //                 Vertices.push_back(points[i]);
        //                 x0 = p.x;
        //                 y0 = p.y;
        //                 ++ num;
        //             }else
        //                 break;
        //         }
        //     }
        // }
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
    //Scene& RefScene = Scenes[ID];
    std::vector<int>& srcID = Scenes[ID].srcID;
    num_img = srcID.size();

    //set images and camera data
    std::string image_folder = input_folder + std::string("/images");
    std::string cam_folder = input_folder + std::string("/cams");   

    for (size_t i = 0; i < num_img; ++i) {
        Scene& Scene = Scenes[srcID[i]];
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << srcID[i] << ".jpg";
        cv::Mat_<uint8_t> img_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
        if(img_uint.empty())  //判断是否有数据
        {   
            std::cout<<"Can not read this image !"<<image_path.str()<<std::endl;  
        }
        img_uint.convertTo(Scene.image,CV_32FC1);
        images.push_back(Scene.image);
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << srcID[i] << "_cam.txt";
        Camera cam = ReadCamera(cam_path.str());
        cam.height = Scene.image.rows;
        cam.width = Scene.image.cols;
        cameras.push_back(cam);
    }
    
    //Adjust image scale
    for (size_t i = 0; i < num_img; ++i) {
        const int idx = srcID[i];
        int max_image_size = Scenes[idx].max_image_size;
        if (images[i].cols <= max_image_size && images[i].rows <= max_image_size) {
            continue;
        }
        const float factor_x = static_cast<float>(max_image_size) / images[i].cols;
        const float factor_y = static_cast<float>(max_image_size) / images[i].rows;
        const float factor = std::min(factor_x, factor_y);

        const int new_cols = std::round(images[i].cols * factor);
        const int new_rows = std::round(images[i].rows * factor);

        const float scale_x = new_cols / static_cast<float>(images[i].cols);
        const float scale_y = new_rows / static_cast<float>(images[i].rows);

        if(Scenes[idx].image.empty())  //判断是否有数据
        {   
            std::cout << "src:" << idx << std::endl;
            std::cout << "Can not read this image !" << idx << std::endl;
        }
        cv::Mat_<float> scaled_image_float;
        cv::resize(Scenes[idx].image, scaled_image_float, cv::Size(new_cols,new_rows), 0, 0, cv::INTER_LINEAR);
        Scenes[idx].image = scaled_image_float.clone();
        images[i] = Scenes[idx].image;

        cameras[i].K[0] *= scale_x;
        cameras[i].K[2] *= scale_x;
        cameras[i].K[4] *= scale_y;
        cameras[i].K[5] *= scale_y;
        cameras[i].height = scaled_image_float.rows;
        cameras[i].width = scaled_image_float.cols;
    }
    std::cout << "Start calculating reference image " << srcID[0] << ". There are "<< num_img - 1 <<" source images can be used."<<std::endl;

    //patchmatch params setting
    params.depth_min = cameras[0].depth_min * 0.6f;
    params.depth_max = cameras[0].depth_max * 1.2f;
    //std::cout << "Set depth range: " << params.depth_min << " to " << params.depth_max << std::endl;
    params.num_images = (int)images.size();

    if (params.geom_consistency) {
        std::cout<<"Geometry consistency optimize start"<<std::endl;
        depths.clear();
        //read depths
        std::string suffix = "/depths.dmb";

        size_t num_src_images = srcID.size();
        for (size_t i = 1; i < num_src_images; ++i) {
            Scene& scene = Scenes[srcID[i]];
            std::stringstream result_path;
            result_path << input_folder << "/MPMVS"  << "/2333_" << std::setw(8) << std::setfill('0') << srcID[i];
            std::string result_folder = result_path.str();
            std::string depth_path = result_folder + suffix;
            readDepthDmb(depth_path, scene.depth);
            depths.push_back(scene.depth);
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
    const size_t wh = cameras[0].width * cameras[0].height;
    checkCudaCall(cudaMalloc((void**)&cudaTextureImages, sizeof(cudaTextureObject_t) * num_img));
    checkCudaCall(cudaMalloc((void**)&cudaCameras, sizeof(Camera) * num_img));
    if(params.geom_consistency){
        checkCudaCall(cudaMalloc((void**)&cudaTextureDepths, sizeof(cudaTextureObject_t) * ( num_img - 1)));
        hostGeomCosts = new float[wh];
        checkCudaCall(cudaMalloc((void**)&cudaGeomCosts, sizeof(float) * wh));
        if(params.geomPlanarPrior){
            // hostGeomCosts = new float[wh];
            // checkCudaCall(cudaMalloc((void**)&cudaGeomCosts, sizeof(float) * wh));
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

    for (int i = 0; i < cameras[0].height; ++i){
        for (int j = 0; j < cameras[0].width; ++j)  {
            const int idx = i * cameras[0].width + j;
            hostPlaneMask[idx] = (unsigned int)masks(i, j);
            if (masks(i, j) > 0) {
                hostPriorPlanes[idx] = PlaneParams[masks(i, j) - 1];
            }
        }
    }
    cudaMemcpy(cudaPriorPlanes, hostPriorPlanes, sizeof(float4) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaPlaneMask, hostPlaneMask, sizeof(unsigned int) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
}

void PatchMatchCUDA::CudaMemInit(Scene &scene){
    for(int i=0; i<num_img; ++i){
        int rows=images[i].rows;
        int cols=images[i].cols;

        const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        checkCudaCall(cudaMallocArray(&cudaImageArrays[i], &channelDesc, cols, rows));
        checkCudaCall(cudaMemcpy2DToArray(cudaImageArrays[i],0,0, images[i].ptr<float>(), images[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice));

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cudaImageArrays[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode  = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        checkCudaCall(cudaCreateTextureObject(&textureImages[i], &resDesc, &texDesc, NULL));
        //cudaError_t cudaMemcpy2DToArray (struct cudaArray *  	dst,size_t  wOffset,size_t  hOffset,const void *  src,size_t  spitch,size_t  width,size_t 	height,enum cudaMemcpyKind 	kind)
    }

    checkCudaCall(cudaMemcpy(cudaTextureImages, textureImages.data(), sizeof(cudaTextureObject_t)*num_img, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(cudaCameras, cameras.data(), sizeof(Camera)*num_img, cudaMemcpyHostToDevice));
    
    if(params.geom_consistency){
        for(int i=0; i<num_img-1; ++i){
            int rows=depths[i].rows;
            int cols=depths[i].cols;
            const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            checkCudaCall(cudaMallocArray(&cudaDepthArrays[i], &channelDesc, cols, rows));
            checkCudaCall(cudaMemcpy2DToArray(cudaDepthArrays[i],0,0, depths[i].ptr<float>(), depths[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice));

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cudaDepthArrays[i];

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
        for (int row = 0; row < height; ++row){
            for (int col = 0; col < width; ++col) {
                const int idx = row * width + col;
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
    if(params.geomPlanarPrior){
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
        delete[] hostGeomCosts;
        cudaFree(cudaGeomCosts);
        cudaFree(cudaTextureDepths);  
        
	}
}

