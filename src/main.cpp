#include <iostream>
#include "PatchMatch.h"
#include "FileIO.h"

using namespace std;

std::vector<Scene> Scenes;
int main(int argc,char *argv[]){
    std::string yaml_path="/home/xuan/MP-MVS/src/config/config.yaml";
    ConfigParams config=readConfig(yaml_path);


    mkdir(config.output_folder.c_str(), 0777);
    //生成视图集合
    GenerateSampleList(config.input_folder, Scenes);
    int num_img=Scenes.size();
    std::cout << "There are " << num_img<< " Depthmap needed to be computed!\n" << std::endl;

    Time time;
    time.start();
    
    for(int i=0;i<num_img;++i){
        //ProcessProblem(config.input_folder,config.output_folder, Scenes, i, false, !config.gp&&config.planar_prior);
    }

    for (int geom_iter = 0; geom_iter < config.geom_iterations; ++geom_iter) {
        if(config.gp&&geom_iter!=config.geom_iterations-1)
            config.planar_prior=true;
        else
            config.planar_prior=false;
        for (size_t i = 0; i < num_img; ++i) {
            ProcessProblem(config.input_folder,config.output_folder, Scenes, i,config.geom_consistency,config.planar_prior);
        }
    }
    printf("cost time is %.10f us\n", time.cost());
    // if(config.sky_seg)
    //     GenerateSkyRegionMask(Scenes,config.input_folder);
    RunFusion(config.input_folder,Scenes,config.sky_seg);
   
    


    // ProcessProblem(input_folder,output_folder, Scenes, 0,geom_consistency,planar_prior);
    // // // cv::Mat_<cv::Vec3f> normal;
    // // // // readNormalDmb("/home/xuan/MP-MVS/dense/MPMVS/2333_00000000/normals.dmb",normal);
    // // // // cv::imwrite("/home/xuan/MP-MVS/result/normal.jpg",normal);
    cv::namedWindow("dmap", (800,1200));
    // // // //cv::imshow("dmap",normal);
    // // // //cv::waitKey(0);
    cv::Mat_<float> dmap,costs;
    // // readDepthDmb("/home/xuan/MP-MVS/dense/MPMVS/2333_00000007/depths.dmb",dmap);
    // // DmbVisualize(dmap,"MPMVS.jpg");
    // // readDepthDmb("/home/xuan/ACMM-main/dense/ACMM/2333_00000007/depths_geom.dmb",dmap);
    // // DmbVisualize(dmap,"ACMM.jpg");
    // // readDepthDmb("/home/xuan/ACMMP-main/dense/ACMMP/2333_00000007/depths_geom.dmb",dmap);
    // // DmbVisualize(dmap,"ACMMP.jpg");
    readDepthDmb("/home/xuan/MP-MVS/dense/MPMVS/2333_00000000/depths_prior.dmb",dmap);
    // // //readDepthDmb("/home/xuan/ACMH-main/dense/ACMH/2333_00000000/costs.dmb",costs);
    // // readColmapDmap("/home/xuan/colmap/data/dense/images/stereo/depth_maps/00000007.jpg.geometric.bin",dmap);
    DmbVisualize(dmap,"COLMAP.jpg");
    // readGT("/home/xuan/MP-MVS/ground_truth_depth/dslr_images/DSC_0634.JPG",dmap);
    //GTVisualize(dmap);
    // if(config.dmap_eval)
    // {
    //     //cv::Mat_<float> dmap;
    //     //readGT("/home/xuan/MP-MVS/ground_truth_depth/dslr_images/DSC_0286.JPG", dmap);
    //     //GTVisualize(dmap);
    //     std::vector<double> score = DmapEval(config.input_folder,config.GT_folder,"/MPMVS","/depths.dmb",0.02);
    // }
    // std::vector<double> score = ColmapEval("/home/xuan/colmap/data/dslr_images_undistorted",config.GT_folder,0.5);






}