#include "main.h"

using namespace std;

std::vector<Scene> Scenes;
int main(int argc,char *argv[]){
    std::string yaml_path="/home/xuan/MP-MVS/config/config.yaml";
    ConfigParams config=readConfig(yaml_path);

    mkdir(config.output_folder.c_str(), 0777);
    GenerateSampleList(config.input_folder, Scenes);
    int num_img = Scenes.size();
    std::cout << "There are " << num_img << " Depthmaps needed to be computed!\n" << std::endl;

    Time time;
    time.start();
    //Multi-Scale windows PatchMatch
    for(int i=0; i<num_img; ++i)
        ProcessProblem(config.input_folder,config.output_folder, Scenes, i, false, !config.geomPlanarPrior && config.planar_prior);

    for (int geom_iter = 0; geom_iter < config.geom_iterations; ++geom_iter) {
        if(config.geomPlanarPrior && geom_iter != config.geom_iterations - 1)
            config.planar_prior = true;
        else
            config.planar_prior = false;
        for (size_t i = 0; i < num_img; ++i) {
            ProcessProblem(config.input_folder, config.output_folder, Scenes, i,true, config.planar_prior);
        }
    }
    printf("cost time is %.10f us\n", time.cost());
    // if(config.sky_seg)
    //     GenerateSkyRegionMask(Scenes,config.input_folder);
    RunFusion(config.input_folder,config.output_folder, Scenes,config.sky_seg);
   

    // cv::Mat_<cv::Vec3f> normal;
    // readNormalDmb("/home/xuan/MP-MVS/dense/MPMVS/2333_00000000/normals.dmb",normal);
    // cv::imwrite("/home/xuan/MP-MVS/result/normal.jpg",normal);
    //cv::namedWindow("dmap", (800,1200));
    //cv::imshow("dmap",normal);
    //cv::waitKey(0);
    cv::Mat_<float> dmap;
    readDepthDmb("/home/xuan/MP-MVS/dense/MPMVS/2333_00000000/depths.dmb",dmap);
    //readColmapDmap("/home/xuan/colmap/data/dense/images/stereo/depth_maps/00000007.jpg.geometric.bin",dmap);
    DmbVisualize(dmap,"1.jpg");
    // readGT("/home/xuan/MP-MVS/ground_truth_depth/dslr_images/DSC_0634.JPG",dmap);
    // GTVisualize(dmap);
}