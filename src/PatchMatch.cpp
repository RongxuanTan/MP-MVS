#include "PatchMatch.h"

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

    for(int i=0;i<3;++i){
        camera.C[i]=-(camera.R[i]*camera.t[0]+camera.R[i+3]*camera.t[1]+camera.R[i+6]*camera.t[2]);
    }

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

void RunFusion(std::string &dense_folder, const std::vector<Scene> &Scenes, bool geom_consistency)
{
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
        // if (geom_consistency) {
        //     suffix = "/depths_geom.dmb";
        // }
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

    // for (size_t i = 0; i < num_images; ++i) {
    //     std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
    //     const int cols = depths[i].cols;
    //     const int rows = depths[i].rows;
    //     int num_ngb = Scenes[i].srcID.size();
    //     std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
    //     for (int r =0; r < rows; ++r) {
    //         for (int c = 0; c < cols; ++c) {
    //             if (masks[i].at<uchar>(r, c) == 1)
    //                 continue;
    //             float ref_depth = depths[i].at<float>(r, c);
    //             cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);

    //             if (ref_depth <= 0.0)
    //                 continue;

    //             float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
    //             float3 consistent_Point = PointX;
    //             cv::Vec3f consistent_normal = ref_normal;
    //             float consistent_Color[3] = {(float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2]};
    //             int num_consistent = 0;
    //             float dynamic_consistency = 0;

    //             for (int j = 0; j < num_ngb; ++j) {
    //                 int src_id = image_id_2_index[Scenes[i].srcID[j]];
    //                 const int src_cols = depths[src_id].cols;
    //                 const int src_rows = depths[src_id].rows;
    //                 float2 point;
    //                 float proj_depth;
    //                 ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
    //                 int src_r = int(point.y + 0.5f);
    //                 int src_c = int(point.x + 0.5f);
    //                 if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
    //                     if (masks[src_id].at<uchar>(src_r, src_c) == 1)
    //                         continue;

    //                     float src_depth = depths[src_id].at<float>(src_r, src_c);
    //                     cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
    //                     if (src_depth <= 0.0)
    //                         continue;

    //                     float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
    //                     float2 tmp_pt;
    //                     ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
    //                     float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
    //                     float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
    //                     float angle = GetAngle(ref_normal, src_normal);

    //                     if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
    //                         used_list[j].x = src_c;
    //                         used_list[j].y = src_r;

    //                         float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
    //                         float cons = exp(-tmp_index);
    //                         dynamic_consistency += exp(-tmp_index);
    //                         num_consistent++;
    //                     }
    //                 }
    //             }

    //             if (num_consistent >= 1 && (dynamic_consistency > 0.3 * num_consistent)) {
    //                 PointList point3D;
    //                 point3D.coord = consistent_Point;
    //                 point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
    //                 point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
    //                 PointCloud.push_back(point3D);

    //                 for (int j = 0; j < num_ngb; ++j) {
    //                     if (used_list[j].x == -1)
    //                         continue;
    //                     masks[image_id_2_index[Scenes[i].srcID[j]]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
    //                 }
    //             }
    //         }
    //     }
    // }
    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        const int cols = depths[i].cols;
        const int rows = depths[i].rows;
        int num_ngb = Scenes[i].srcID.size();
        std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
        for (int r =0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (masks[i].at<uchar>(r, c) == 1)
                    continue;
                const float& ref_depth = depths[i].at<float>(r, c);
                const cv::Vec3f& ref_normal = normals[i].at<cv::Vec3f>(r, c);

                if (ref_depth <= 0.0)
                    continue;

                float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
                float3 consistent_Point = PointX;
                cv::Vec3f consistent_normal = ref_normal;
                float consistent_Color[3] = {(float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2]};
                int num_consistent = 0;

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
                        float reproj_error = pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2);
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float angle = GetAngle(ref_normal, src_normal);

                        if (reproj_error < 4.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
                            consistent_Point.x += tmp_X.x;
                            consistent_Point.y += tmp_X.y;
                            consistent_Point.z += tmp_X.z;
                            consistent_normal = consistent_normal + src_normal;
                            consistent_Color[0] += images[src_id].at<cv::Vec3b>(src_r, src_c)[0];
                            consistent_Color[1] += images[src_id].at<cv::Vec3b>(src_r, src_c)[1];
                            consistent_Color[2] += images[src_id].at<cv::Vec3b>(src_r, src_c)[2];

                            used_list[j].x = src_c;
                            used_list[j].y = src_r;
                            num_consistent++;
                        }
                    }
                }

                if (num_consistent >= 2) {
                    consistent_Point.x /= (num_consistent + 1.0f);
                    consistent_Point.y /= (num_consistent + 1.0f);
                    consistent_Point.z /= (num_consistent + 1.0f);
                    consistent_normal /= (num_consistent + 1.0f);
                    consistent_Color[0] /= (num_consistent + 1.0f);
                    consistent_Color[1] /= (num_consistent + 1.0f);
                    consistent_Color[2] /= (num_consistent + 1.0f);

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

void ProcessProblem(const std::string &input_folder,const std::string &output_folder, std::vector<Scene> &Scenes, const int ID,bool geom_consistency=false){
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
    MP.SetGeomConsistencyParams(geom_consistency);
    MP.PatchMatchInit(Scenes,ID);
    MP.AllocatePatchMatch();
    MP.CudaMemInit(Scenes[ID]);
    MP.Run();

    const int width = MP.GetReferenceImageWidth();
    const int height = MP.GetReferenceImageHeight();

    cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);
    int count=0;
    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            float4 plane_hypothesis = MP.GetPlaneHypothesis(center);
            depths(row, col) = plane_hypothesis.w;
            normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            costs(row, col) = MP.GetCost(center);
        }
    }
    std::string suffix = "/depths.dmb";
    // if (geom_consistency) {
    //     suffix = "/depths_geom.dmb";
    // }
    std::string depth_path = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path = result_folder + "/costs.dmb";
    writeDepthDmb(depth_path, depths);
    writeNormalDmb(normal_path, normals);
    writeDepthDmb(cost_path, costs);
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << scene.refID << " done!" << std::endl;

    MP.Release(Scenes,ID);
}

void PatchMatchCUDA::SetGeomConsistencyParams(bool geom_consistency)
{
    params.geom_consistency = geom_consistency;
    if(geom_consistency)
        params.max_iterations = 2;
}

float4 PatchMatchCUDA::GetPlaneHypothesis(const int index)
{
    return hostPlaneHypotheses[index];
}

float PatchMatchCUDA::GetCost(const int index)
{
    return hostCosts[index];
}

int PatchMatchCUDA::GetReferenceImageWidth()
{
    return cameras[0].width;
}

int PatchMatchCUDA::GetReferenceImageHeight()
{
    return cameras[0].height;
}

void PatchMatchCUDA::SetFolder(const std::string &_input_folder,const std::string &_output_folder){
    input_folder=_input_folder;
    output_folder=_output_folder;
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
    int max_image_size = scene.cur_image_size;
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
        max_image_size = Scenes[srcID].cur_image_size;
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
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";

        std::string depth_path = result_folder + suffix;
        readDepthDmb(depth_path, scene.depth);
        depths.push_back(scene.depth);

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
        cudaDepthArrays.resize(num_img);
        textureDepths.resize(num_img);     
    }
}

void PatchMatchCUDA::AllocatePatchMatch(){
    checkCudaCall(cudaMalloc((void**)&cudaTextureImages, sizeof(cudaTextureObject_t) * num_img));
    checkCudaCall(cudaMalloc((void**)&cudaCameras, sizeof(Camera) * num_img));
    if(params.geom_consistency){
        checkCudaCall(cudaMalloc((void**)&cudaTextureDepths, sizeof(cudaTextureObject_t) * num_img));
    }

    const size_t wh=cameras[0].width*cameras[0].height;
    //平面假设
    hostPlaneHypotheses = new float4[wh];
    checkCudaCall(cudaMalloc((void**)&cudaPlaneHypotheses, sizeof(float4) * wh));
    //纹理系数图像
    checkCudaCall(cudaMalloc((void**)&cudaTexCofMap, sizeof(uchar) * wh));
    hostCosts = new float[wh];
    checkCudaCall(cudaMalloc((void**)&cudaCosts, sizeof(float) * wh));
    checkCudaCall(cudaMalloc((void**)&cudaPreCosts, sizeof(float) * wh));
    checkCudaCall(cudaMalloc((void**)&cudaRandStates, sizeof(curandState) * wh));
    checkCudaCall(cudaMalloc((void**)&cudaSelectedViews, sizeof(unsigned int) * wh));
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
        for(int i=0;i<num_img;++i){
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
        checkCudaCall(cudaMemcpy(cudaTextureDepths, textureDepths.data(), sizeof(cudaTextureObject_t)*num_img, cudaMemcpyHostToDevice));
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

        if(ref_depth.empty())  //判断是否有数据
        {   
            std::cout<<"Can not read this depth image !"<<std::endl; 
            exit(0); 
        }
        std::cout<<"cols:"<<ref_depth.cols<<std::endl;
        
        int width = ref_depth.cols;
        int height = ref_depth.rows;
        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int center = row * width + col;
                float4 PlaneHypothese;
                PlaneHypothese.x = ref_normal(row, col)[0];
                PlaneHypothese.y = ref_normal(row, col)[1];
                PlaneHypothese.z = ref_normal(row, col)[2];
                PlaneHypothese.w = ref_depth.at<float>(row, col);
                hostPlaneHypotheses[center] = PlaneHypothese;
                hostCosts[center] = ref_cost(row, col);
            }
        }
        checkCudaCall(cudaMemcpy(cudaPlaneHypotheses, hostPlaneHypotheses, sizeof(float4)*width*height, cudaMemcpyHostToDevice));
        checkCudaCall(cudaMemcpy(cudaCosts, hostCosts, sizeof(float)*width*height, cudaMemcpyHostToDevice));
    }

}

void PatchMatchCUDA::Release(std::vector<Scene> Scenes,const int &ID)
{
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
    cudaFree(cudaPreCosts);
    cudaFree(cudaRandStates);
    cudaFree(cudaSelectedViews);
    cudaFree(cudaTexCofMap);

    if(params.geom_consistency){
        scene.depth.release();
        for(auto id:srcIDs){
            Scene& Srcscene = Scenes[id];
            Srcscene.depth.release();
        }
        for(int i=0;i<num_img;++i) 
        {
           cudaDestroyTextureObject(textureDepths[i]);
		   cudaFreeArray(cudaDepthArrays[i]); 
        } 
        cudaFree(cudaTextureDepths);  
	}
}

