#include <iostream>
#include "PatchMatch.h"
#include "FileIO.h"
#include "cmdline.h"
using namespace std;

std::vector<Scene> Scenes;
int main(int argc,char *argv[]){
    cmdline::parser a;
    a.add<string>("input-folder",'i',"input data path",true,"");
    a.add<string>("output-folder",'o',"output data path",false,"");
    a.parse_check(argc,argv);

    std::string input_folder=a.get<string>("input-folder");
    std::string output_folder=a.get<string>("output-folder");
    if(output_folder==""){
        checkpath(input_folder);
        output_folder=input_folder+"/MPMVS";
    }else
    checkpath(output_folder);
    
    cout<<"Input data path:"<<input_folder<<endl;
    cout<<"Output data path:"<<output_folder<<endl;
    mkdir(output_folder.c_str(), 0777);
    //生成视图集合
    GenerateSampleList(input_folder, Scenes);
    int num_img=Scenes.size();
    std::cout << "There are " << num_img<< " Depthmap needed to be computed!\n" << std::endl;

    
    int flag = 0;
    int geom_iterations = 2;
    bool geom_consistency = false;
    for(int i=0;i<num_img;++i){
        ProcessProblem(input_folder,output_folder, Scenes, i,geom_consistency);

    }
    geom_consistency = true;
    for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
        for (size_t i = 0; i < num_img; ++i) {
            ProcessProblem(input_folder,output_folder, Scenes, i,geom_consistency);
        }
    }
    RunFusion(input_folder,Scenes,geom_consistency);

    //ProcessProblem(input_folder,output_folder, Scenes, 5,geom_consistency);
    // cv::Mat_<cv::Vec3f> normal;
    // readNormalDmb("/home/xuan/MP-MVS/dense/MPMVS/2333_00000005/normals.dmb",normal);
    // //cv::imwrite("/home/xuan/MP-MVS/result/normal.jpg",normal);
    // cv::namedWindow("dmap", (800,1200));
    // cv::imshow("dmap",normal);
    // cv::waitKey(0);
    // cv::Mat_<float> dmap,costs,dmapg;
    // readDepthDmb("/home/xuan/MP-MVS/dense/MPMVS/2333_00000005/depths.dmb",dmap);
    // // readDepthDmb("/home/xuan/MP-MVS/dense/MPMVS/2333_00000005/depths_geom.dmb",dmapg);
    // readDepthDmb("/home/xuan/ACMH-main/dense/ACMH/2333_00000005/costs.dmb",costs);
    // DmbVisualize(dmap);
    // DmbVisualize(dmapg);

    //深度图转ply
    // std::vector<PointList> pc;
    // cv::Mat color=cv::imread("/home/xuan/MP-MVS/dense/images/00000005.jpg",1);
    // float fx=3409.58,fy=3409.44,cx=3115.16 ,cy=2064.73;
    // const int rows=dmap.rows;
    // const int cols=dmap.cols;
    // for(int i=0;i<rows;++i){
    //     for(int j=0;j<cols;++j){
    //         const float cost=costs.at<float>(i,j);
    //         if(cost<0.2f){
    //             float const depth=dmap.at<float>(i,j);
    //             PointList p;
    //             p.coord.x=depth * ((float)j - cx) / fx;
    //             p.coord.y=depth * ((float)i - cy) / fy;
    //             p.coord.z=depth;
    //             p.normal=make_float3(0,0,0);
    //             p.color=make_float3((float)color.at<cv::Vec3b>(i, j)[0],(float)color.at<cv::Vec3b>(i, j)[1],(float)color.at<cv::Vec3b>(i, j)[2]);
    //             pc.push_back(p);
    //         }

    //     }
    // }
    // StoreColorPlyFileBinaryPointCloud ("/home/xuan/MP-MVS/1.ply", pc);




}