#include "main.h"
using namespace std;

std::vector<Scene> Scenes;
std::string project_path = PROJECT_PATH;
int main(int argc,char *argv[]) {
    std::string yaml_path = project_path + "/config/config.yaml";
    ConfigParams config = readConfig(yaml_path);

    mkdir(config.output_folder.c_str(), 0777);
    GenerateSampleList(config, Scenes);

    int num_img = Scenes.size();
    std::cout << "There are " << num_img << " depthmaps need to be computed!\n" << std::endl;

    Time time;
    time.start();
    //Multi-Scale windows PatchMatch
    bool planar_prior = !config.geomPlanarPrior && config.planar_prior;
    for(int i = 0; i < num_img; ++i)
        ProcessProblem(config.input_folder, config.output_folder, Scenes, i, false, planar_prior);

    planar_prior = config.planar_prior;
    for (int geom_iter = 0; geom_iter < config.geom_iterations; ++geom_iter) {
        if(config.geomPlanarPrior && geom_iter != config.geom_iterations - 1)
            planar_prior = true;
        else
            planar_prior = false;
        for (size_t i = 0; i < num_img; ++i) {
            ProcessProblem(config.input_folder, config.output_folder, Scenes, i,true, planar_prior);
        }
    }
    printf("cost time is %.10f us\n", time.cost());
#ifdef BUILD_NCNN
    if(config.sky_seg)
        GenerateSkyRegionMask(Scenes, project_path, config);
#endif

    RunFusion(config, Scenes);

    if (config.saveDmb || config.saveProirDmb || config.saveCostDmb || config.saveNormalDmb) {
        saveDmbAsJpg(config, num_img, true);
    }

}