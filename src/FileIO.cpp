#include "FileIO.h"
#include <Eigen/Dense>

void checkpath(std::string &path){
    if(path.back()=='/')
        path.pop_back();
}

ConfigParams readConfig(std::string yaml_path){
    ConfigParams config;
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);

    fs["Input-folder"]>>config.input_folder;
    fs["Output-folder"]>>config.output_folder;
    fs["Geometric consistency iterations"]>>config.geom_iterations;
    fs["Planer prior"]>>config.planar_prior;
    fs["Geometric consistency planer prior"]>>config.gp;
    fs["Sky segment"]>>config.sky_seg;
    //fs["depth map eval"]>>config.dmap_eval;
    //fs["depth map eval folder"]>>config.GT_folder;

    checkpath(config.input_folder);
    if(config.output_folder!=" "){
        checkpath(config.output_folder);
    }else
    config.output_folder=config.input_folder+"/MPMVS";
    std::cout<<"Input data path:"<<config.input_folder<<std::endl;
    std::cout<<"Output data path:"<<config.output_folder<<std::endl;

    return config;
}

bool readGT(const std::string file_path, cv::Mat_<float> &depth)
{
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage){
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t h=4032, w=6048, nb=1;
    int32_t dataSize = h*w*nb;

    depth = cv::Mat::zeros(h,w,CV_32F);
    fread(depth.data,sizeof(float),dataSize,inimage);

    fclose(inimage);
    return true;
}

bool GTVisualize(cv::Mat_<float> &depth)
{
    if(depth.empty())  //判断是否有数据
    {   
        std::cout<<"Can not read this depth image !"<<std::endl;  
        exit(0);
    }
    float max=0,min=1000;
    const int rows=depth.rows;
    const int cols=depth.cols;
    cv::Mat mask = cv::Mat::ones(depth.size(), CV_8UC1);
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            const float val=depth.at<float>(i,j);
            if(val<1000&&val>0){
                if(val>max)
                    max=val;
                if(val<min)
                    min=val;
                mask.at<uchar>(i,j)=255;
            }else
            mask.at<uchar>(i,j)=0;
        }
    }
    std::cout<<"min:"<<min<<"  max:"<<max<<std::endl;

    double inv_max=1/(max-min+1e-8);    
    cv::Mat norm_dmb=depth-min;
    norm_dmb=norm_dmb*inv_max;

    cv::Mat uint_dmb,color_dmb;
    int32_t num_min=0,num_max=0;
    norm_dmb.convertTo(uint_dmb,CV_8UC3,255.0);
    // cv::namedWindow("dmap", (800,1200));
    // cv::imshow("dmap",uint_dmb);
    // cv::waitKey(0); 
    cv::Mat bgr_img=cv::Mat::zeros(rows,cols,CV_8UC3);    
    cv::applyColorMap(uint_dmb,color_dmb,cv::COLORMAP_JET);
    
    Colormap2Bgr(color_dmb,bgr_img,mask);
    cv::imwrite("/home/xuan/MP-MVS/result/GT.jpg",bgr_img);

    cv::namedWindow("dmap", (800,1200));
    cv::imshow("dmap",bgr_img);
    cv::waitKey(0);   

}

void GetFileNames(std::string path,std::vector<std::string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;

    if(!(pDir = opendir(path.c_str()))){
        std::cout<<"Folder doesn't Exist!"<<std::endl;
        return;
    }

    while((ptr = readdir(pDir))!=NULL) {
        if (strcmp(ptr->d_name, ".") !=NULL  && strcmp(ptr->d_name, "..") != NULL){
            filenames.push_back(path + "/" + ptr->d_name);
    }
    }
    closedir(pDir);

}
void GetSubFileNames(std::string path,std::vector<std::string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;

    if(!(pDir = opendir(path.c_str()))){
        std::cout<<"Folder doesn't Exist!"<<std::endl;
        return;
    }

    while((ptr = readdir(pDir))!=NULL) {
        if (strcmp(ptr->d_name, ".") !=NULL  && strcmp(ptr->d_name, "..") != NULL){
            filenames.push_back(ptr->d_name);
    }
    // for(int i=0;i<filenames.size();++i){
    //     filenames[i]=filenames[i].substr(0,8);
    // }
    }
    closedir(pDir);

}

std::vector<double> DmapEval(const std::string data_folder,const std::string GT_folder,const std::string method_name ,const std::string depth_name,float error = 0.02){
    std::vector<std::string> GT_path;
    std::vector<double> score;
    GetFileNames(GT_folder, GT_path);
    sort(GT_path.begin(), GT_path.end());
    for(int i=0;i<GT_path.size();++i){
        std::cout<<GT_path[i]<<std::endl;
    }

    int num_images;
    {
        std::string cluster_list_path = data_folder + std::string("/pair.txt");
        std::ifstream file(cluster_list_path);
        if(!file.is_open()){
            std::cout<<"can not open file in path:   "<<cluster_list_path<<std::endl;
            exit(1);
        }
        file >> num_images;
    }

    if(num_images==GT_path.size())
        std::cout<<"start depth map evalution"<<std::endl;
    else{
        std::cout<<"nums of dmap is not equal to nums of GT!!!"<<num_images<<" "<<GT_path.size()<<std::endl;
    }
    long all_images_p=0,all_right_p=0;
    double s=0;
    cv::Mat_<float> depth,GT,temp;
    
    for(int i=num_images-1;i<num_images;++i){
        std::stringstream result_path;
        result_path << data_folder << method_name << "/2333_" << std::setw(8) << std::setfill('0') << i;
        std::string result_folder = result_path.str();
        std::string suffix = depth_name;
        std::string depth_path = result_folder + suffix;
        readDepthDmb(depth_path, depth);
        //DmbVisualize(depth);
        readGT(GT_path[i],GT);
        //GTVisualize(GT);

        const int rows=GT.rows;
        const int cols=GT.cols;
        cv::Mat mask = cv::Mat::ones(GT.size(), CV_8UC1);
        const float sx=(float)depth.rows/(float)GT.rows;
        const float sy=(float)depth.cols/(float)GT.cols;
        long images_p=0,right_p=0;
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                const float dt=GT.at<float>(i,j);
                if(dt<100.0f){
                    ++images_p;
                    const float d=depth.at<float>((float)i*sx+0.5,(float)j*sy+0.5);
                    if(abs(dt-d)<error){
                        //std::cout<<dt<<" "<<d<<std::endl;
                        ++right_p;
                        mask.at<uchar>(i,j)=255;
                    }else
                    mask.at<uchar>(i,j)=100;
                }else
                mask.at<uchar>(i,j)=0;
            }
        }
        all_images_p=all_images_p+images_p;
        all_right_p=all_right_p+right_p;

        std::cout<<images_p<<" "<<right_p<<std::endl;
        score.push_back((double)right_p/(double)images_p);
        std::cout<<"image "<<i<<" score:"<<score[i]<<std::endl;
        s=s+(double)right_p/(double)images_p;
        cv::imshow("1",mask);
        cv::waitKey(0);
    }
    //     for(int i=0;i<num_images;++i){
    //     std::stringstream result_path;
    //     result_path << data_folder << method_name << "/2333_" << std::setw(8) << std::setfill('0') << i;
    //     std::string result_folder = result_path.str();
    //     std::string suffix = depth_name;
    //     std::string depth_path = result_folder + suffix;
    //     readDepthDmb(depth_path, depth);
    //     //DmbVisualize(depth);
    //     readGT(GT_path[i],GT);
    //     //GTVisualize(GT);

    //     const int rows=depth.rows;
    //     const int cols=depth.cols;
    //     const float sx=(float)GT.rows/(float)depth.rows;
    //     const float sy=(float)GT.cols/(float)depth.cols;
    //     long images_p=0,right_p=0;
    //     for(int i=0;i<rows;++i){
    //         for(int j=0;j<cols;++j){
    //             const float dt=GT.at<float>((float)i*sx+0.5,(float)j*sy+0.5);
    //             const float d=depth.at<float>(i,j);
    //             if(dt<1000.0f){
    //                 ++images_p;
    //                 if(abs(dt-d)<error){
    //                     //std::cout<<dt<<" "<<d<<std::endl;
    //                     ++right_p;
    //                 }
    //             }
    //         }
    //     }
    //     all_images_p=all_images_p+images_p;
    //     all_right_p=all_right_p+right_p;

    //     std::cout<<images_p<<" "<<right_p<<std::endl;
    //     score.push_back((double)right_p/(double)images_p);
    //     std::cout<<"image "<<i<<" score:"<<score[i]<<std::endl;
    //     s=s+(double)right_p/(double)images_p;
    // }
    std::cout<<all_images_p<<" "<<all_right_p<<std::endl;
    //std::cout<<"all:"<<((double)all_right_p/(double)all_images_p)<<std::endl;
    std::cout<<"all:"<<s/num_images<<std::endl;
 
    return {};

}

bool readColmapDmap(const std::string file_path, cv::Mat_<float> &depth)
{
    std::fstream text_file(file_path, std::ios::in | std::ios::binary);

    size_t width = 0;
    size_t height = 0;
    size_t d= 0;
    char unused_char;
    text_file >> width >> unused_char >> height >> unused_char >> d >>unused_char;
    //std::cout<<width<<" "<<height<<" "<<d<<std::endl;
    const std::streampos pos = text_file.tellg();
    text_file.close();

    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage){
        std::cout << "Error opening file " << file_path << std::endl;
        return {};
    }

    int32_t h, w, nb;


    fread(&h,sizeof(int32_t),1,inimage);
    fread(&w,sizeof(int32_t),1,inimage);
    fread(&nb,sizeof(int32_t),1,inimage);

    int32_t dataSize = (int32_t)height*(int32_t)width*(int32_t)d;

    depth = cv::Mat::zeros(height,width,CV_32F);
    fread(depth.data,sizeof(float),dataSize,inimage);

    fclose(inimage);
    //DmbVisualize(depth);
}

std::vector<double> ColmapEval(const std::string data_folder,const std::string GT_folder,float error = 0.02){
    std::vector<std::string> dmap_path;
    std::vector<double> score;
    GetSubFileNames(data_folder+"/images", dmap_path);
    sort(dmap_path.begin(), dmap_path.end());
    for(int i=0;i<dmap_path.size();++i){
        std::cout<<dmap_path[i]<<std::endl;
    }
    long all_images_p=0,all_right_p=0;
    double s=0,n=0;
    cv::Mat_<float> depth,GT,temp;
    int num_images=dmap_path.size();
    // double k1,k2;
    // {
    //     Eigen::Matrix3d R1,R2;
    // Eigen::Isometry3d T1=Eigen::Isometry3d::Identity(),T2=Eigen::Isometry3d::Identity();
    // Eigen::Quaterniond Q1(0.99231581211952313,0.002187045250814988,-0.12371154279477672,-5.4752513097291406e-06);
    // Eigen::Quaterniond Q2(0.99984412277752088,0.0025563922640812575,-0.017411297113732009,-0.001428894208595863);//w x y z
    // //Qua_w_o.normalize();
    // R1= Q1.matrix();
    // T1.rotate(R1);
    // T1.pretranslate ( Eigen::Vector3d ( 4.3480378441731418,-0.030393338225296621,0.51961185092650741 ) ); 
    // R2= Q2.matrix();
    // T2.rotate(R2);
    // T2.pretranslate ( Eigen::Vector3d (1.845696204857737,-0.026299275849290659,-0.24450576388533432) ); 
    // Eigen::Matrix4d T3=T1.matrix().inverse()*T2.matrix();
    // std::cout<<T3<<std::endl;
    // std::cout<<sqrt(T3(0,3)*T3(0,3)+T3(1,3)*T3(1,3)+T3(2,3)*T3(2,3))<<std::endl;
    // k1=sqrt(T3(0,3)*T3(0,3)+T3(1,3)*T3(1,3)+T3(2,3)*T3(2,3));
    // }

    // {
    // Eigen::Matrix3d R1,R2;
    // Eigen::Isometry3d T1=Eigen::Isometry3d::Identity(),T2=Eigen::Isometry3d::Identity();
    // Eigen::Quaterniond Q1(0.714118,0.695542,-0.0574465,0.0543732);
    // Eigen::Quaterniond Q2(0.715931,0.697633,0.0176306,-0.0209686);//w x y z
    // //Qua_w_o.normalize();
    // R1= Q1.matrix();
    // T1.rotate(R1);
    // T1.pretranslate ( Eigen::Vector3d ( 0.841517,0.181774,0.202188 ) ); 
    // R2= Q2.matrix();
    // T2.rotate(R2);
    // T2.pretranslate ( Eigen::Vector3d (0.291006,0.181953,0.0599135) ); 
    // Eigen::Matrix4d T3=T1.matrix().inverse()*T2.matrix();
    // std::cout<<T3<<std::endl;
    // std::cout<<sqrt(T3(0,3)*T3(0,3)+T3(1,3)*T3(1,3)+T3(2,3)*T3(2,3))<<std::endl;
    // k2=sqrt(T3(0,3)*T3(0,3)+T3(1,3)*T3(1,3)+T3(2,3)*T3(2,3));
    // }

    


    for(int i=0;i<num_images;++i){
        std::string dmap_path2 = data_folder+"/stereo/depth_maps/"+dmap_path[i]+".geometric.bin";
        std::string GT_path = GT_folder + "/"+dmap_path[i];
        readColmapDmap(dmap_path2,depth);
        //DmbVisualize(depth);
        //GTVisualize(depth);
        readGT(GT_path,GT);
        //GTVisualize(GT);
        const int drows=depth.rows;
        const int dcols=depth.cols;
        const float dsx=(float)GT.rows/(float)depth.rows;
        const float dsy=(float)GT.cols/(float)depth.cols;
        double d1,dt1;
        for(int i=0;i<drows;++i){
            for(int j=0;j<dcols;++j){
                const float d=depth.at<float>(i,j);
                const float dt=GT.at<float>((float)i*dsx,(float)j*dsy);
                if(d>0&&d<24&&dt<1000){
                    d1=d1+d;
                    dt1=dt1+dt;
                }
                

            }
        }
        
        float k=dt1/d1;
        //k=k2/k1;
        std::cout<<"k:"<<k<<" "<<d1<<std::endl;

        const int rows=GT.rows;
        const int cols=GT.cols;
        const float sx=(float)depth.rows/(float)GT.rows;
        const float sy=(float)depth.cols/(float)GT.cols;
        long images_p=0,right_p=0;
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                const float dt=GT.at<float>(i,j);
                if(dt<1000.0f){
                    ++images_p;
                    const float d=depth.at<float>((float)i*sx,(float)j*sy);
                    if(abs(dt-k*d)<error){
                        //std::cout<<dt<<" "<<d<<std::endl;
                        ++right_p;
                    }
                }
            }
        }
        all_images_p=all_images_p+images_p;
        all_right_p=all_right_p+right_p;

        std::cout<<images_p<<" "<<right_p<<std::endl;
        score.push_back((double)right_p/(double)images_p);
        std::cout<<"image "<<i<<" score:"<<score[i]<<std::endl;
        if((double)right_p/(double)images_p>0.1){
            s=s+(double)right_p/(double)images_p;
            n=n+1;
        }
        
    }
    std::cout<<all_images_p<<" "<<all_right_p<<std::endl;
    //std::cout<<"all:"<<((double)all_right_p/(double)all_images_p)<<std::endl;
    std::cout<<"all:"<<s/n<<std::endl;
 
    return {};





    // cv::Mat_<float> depth1;
    // readColmapDmap(data_folder,depth1);
    // DmbVisualize(depth1);


}

bool readDepthDmb(const std::string file_path, cv::Mat_<float> &depth)
{
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage){
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t),1,inimage);
    fread(&h,sizeof(int32_t),1,inimage);
    fread(&w,sizeof(int32_t),1,inimage);
    fread(&nb,sizeof(int32_t),1,inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h*w*nb;

    depth = cv::Mat::zeros(h,w,CV_32F);
    fread(depth.data,sizeof(float),dataSize,inimage);

    fclose(inimage);
    return true;
}

int writeDepthDmb(const std::string file_path, const cv::Mat_<float> &depth)
{
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");

    int32_t type = 1;
    int32_t h = depth.rows;
    int32_t w = depth.cols;
    int32_t nb = 1;

    fwrite(&type,sizeof(int32_t),1,outimage);
    fwrite(&h,sizeof(int32_t),1,outimage);
    fwrite(&w,sizeof(int32_t),1,outimage);
    fwrite(&nb,sizeof(int32_t),1,outimage);

    float* data = (float*)depth.data;

    int32_t datasize = w*h*nb;
    fwrite(data,sizeof(float),datasize,outimage);

    fclose(outimage);
    return 0;
}


bool readNormalDmb (const std::string file_path, cv::Mat_<cv::Vec3f> &normal)
{
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage) {
        std::cout << "Error opening file " << file_path << std::endl;
        return false;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t),1,inimage);
    fread(&h,sizeof(int32_t),1,inimage);
    fread(&w,sizeof(int32_t),1,inimage);
    fread(&nb,sizeof(int32_t),1,inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h*w*nb;

    normal = cv::Mat::zeros(h,w,CV_32FC3);
    fread(normal.data,sizeof(float),dataSize,inimage);

    fclose(inimage);
    return true;

}

int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> &normal)
{
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");
    if (!outimage) {
        std::cout << "Error opening file " << file_path << std::endl;
    }

    int32_t type = 1; //float
    int32_t h = normal.rows;
    int32_t w = normal.cols;
    int32_t nb = 3;

    fwrite(&type,sizeof(int32_t),1,outimage);
    fwrite(&h,sizeof(int32_t),1,outimage);
    fwrite(&w,sizeof(int32_t),1,outimage);
    fwrite(&nb,sizeof(int32_t),1,outimage);

    float* data = (float*)normal.data;

    int32_t datasize = w*h*nb;
    fwrite(data,sizeof(float),datasize,outimage);

    fclose(outimage);
    return 0;
}

void NormalVisualize(float4 *Plane,const int width,const int height){
    cv::Mat normal(height, width, CV_32FC3);;
    for(int i=0;i<height;++i){
    for(int j=0;j<width;++j){
        const int wh=i * width + j;
        normal.at<cv::Vec3f>(i, j)[0]=Plane[wh].x;
        normal.at<cv::Vec3f>(i, j)[1]=Plane[wh].y;
        normal.at<cv::Vec3f>(i, j)[2]=Plane[wh].z;
        }  
    }
    std::cout<<normal.at<cv::Vec3f>(0, 0)<<std::endl;
    cv::namedWindow("normal", (800,1200));
    cv::imwrite("/home/xuan/MP-MVS/result/normal.jpg",normal);
    cv::imshow("normal",normal);
    cv::waitKey(0);
}


void GetHist(cv::Mat gray,cv::Mat &Hist)    //统计8Bit量化图像的灰度直方图
{
    const int channels[1] = { 0 }; //通道索引
    float inRanges[2] = { 0,255 };  //像素范围
    const float* ranges[1] = {inRanges};//像素灰度级范围
    const int bins[1] = { 256 };   //直方图的维度
    cv::calcHist(&gray, 1, channels,cv::Mat(), Hist,1, bins, ranges);
    //std::cout<<Hist<<std::endl;
    //ShowHist(Hist);
}


void ShowHist(cv::Mat &Hist)
{
    //准备绘制直方图
    int hist_w = 512;
    int hist_h = 400;
    int width = 2;
    cv::Mat histImage = cv::Mat::zeros(hist_h,hist_w,CV_8UC3);   //准备histImage为全黑背景色，大小为512*400
    for (int i = 0; i < Hist.rows; i++)
    {
        cv::rectangle(histImage,cv::Point(width*(i),hist_h-1),cv::Point(width*(i+1),hist_h-cvRound(Hist.at<float>(i)/20)),
        cv::Scalar(255,255,255),-1);//cvRound取整数
    }
    cv::imwrite("/home/xuan/MP-MVS/result/hist.jpg",histImage);
    cv::namedWindow("histImage", cv::WINDOW_AUTOSIZE);
    cv::imshow("histImage", histImage);
    cv::waitKey(0);
}

int getTop10(cv::Mat Hist,int allnum){
    int num=0;
    for(int i=1;i<256;++i)
    {
        num+=(int)Hist.at<float>(i);
        if(((float)num/(float)allnum)>0.03)
            return i-1;
    }
    return 1;
}

int getDown10(cv::Mat Hist,int allnum){
    int num=0;
    for(int i=255;i>1;--i)
    {
        num+=(int)Hist.at<float>(i);
        if(((float)num/(float)allnum)>0.03)
            return i+1;
    }
    return 1;
}

void Colormap2Bgr(cv::Mat &src,cv::Mat &dst,cv::Mat &mask){
    const int rows=src.rows;
    const int cols=src.cols;
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            if(mask.at<uchar>(i,j)!=0)
                dst.at<cv::Vec3b>(i, j)=src.at<cv::Vec3b>(i, j);
            else{
                dst.at<cv::Vec3b>(i, j)[0]=0;
                dst.at<cv::Vec3b>(i, j)[1]=0;
                dst.at<cv::Vec3b>(i, j)[2]=0;
            }  
        }
    }
}

bool DmbVisualize(cv::Mat_<float> &depth,const std::string name)
{
    if(depth.empty())  //判断是否有数据
    {   
        std::cout<<"Can not read this depth image !"<<std::endl;  
        exit(0);
    }
    double max,min=0,real_min=1000,real_max=20;
    const int rows=depth.rows;
    const int cols=depth.cols;
    cv::Mat mask = cv::Mat::ones(depth.size(), CV_8UC1);
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            const float val=depth.at<float>(i,j);
            if(val>110)
            depth.at<float>(i,j)=110;
            if(val>0)
                if(val<real_min)
                    real_min=(double)val;
        }
    }
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            const float val=depth.at<float>(i,j);
            if(val<=0.0){
                mask.at<uchar>(i,j)=0;
                depth.at<float>(i,j)=real_min;
            }
            else{
                mask.at<uchar>(i,j)=255;
             }
        }
    }
    cv::Point maxLoc;
    cv::Point minLoc;
    cv::minMaxLoc(depth, &min, &max, &minLoc, &maxLoc);
    std::cout<<"min:"<<min<<"  max:"<<max<<std::endl;

    double inv_max=1/(max-min+1e-8);    
    cv::Mat norm_dmb=depth-min;
    norm_dmb=norm_dmb*inv_max;

    cv::Mat uint_dmb;
    int32_t num_min=0,num_max=0;
    norm_dmb.convertTo(uint_dmb,CV_8UC3,255.0);
    // cv::namedWindow("dmap", (800,1200));
    // cv::imshow("dmap",uint_dmb);
    // cv::waitKey(0); 
    // cv::Mat bgr_img=cv::Mat::zeros(rows,cols,CV_8UC3);    
    // cv::applyColorMap(uint_dmb,color_dmb,cv::COLORMAP_JET);
    
    // Colormap2Bgr(color_dmb,bgr_img,mask);
    // cv::imwrite("/home/xuan/MP-MVS/result/dmb.jpg",bgr_img);

    // cv::namedWindow("dmap", (800,1200));
    // cv::imshow("dmap",bgr_img);
    // cv::waitKey(0);   

    //使用直方图增加对比度
    cv::Mat Hist;
    GetHist(uint_dmb,Hist);
    int top=getTop10(Hist,rows*cols);
    int down=getDown10(Hist,rows*cols);
    std::cout<<"top:"<<top<<std::endl;
    std::cout<<"down:"<<down<<std::endl;
    double new_min=real_min+(max-real_min)*((float)top/256.0);
    double new_max=real_min+(max-real_min)*((float)down/256.0);
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            const float val=depth.at<float>(i,j);
            if(val<new_min){
                depth.at<float>(i,j)=new_min;
            }
            if(val>new_max)
                depth.at<float>(i,j)=new_max;
        }
    }
    inv_max=1/(new_max-new_min+1e-8); 
    norm_dmb=depth-new_min+1e-8;
    norm_dmb=norm_dmb*inv_max;
    norm_dmb.convertTo(uint_dmb,CV_8UC3,255.0);
    cv::Mat temp;
    cv::applyColorMap(uint_dmb,temp,cv::COLORMAP_JET);
    cv::Mat bgr_img2=cv::Mat::zeros(rows,cols,CV_8UC3); 
    Colormap2Bgr(temp,bgr_img2,mask);
    cv::imwrite("/home/xuan/MP-MVS/result/"+name,bgr_img2);
    cv::namedWindow("dmap", (800,1200));
    cv::imshow("dmap",bgr_img2);
    cv::waitKey(0);   

}

bool CostVisualize(cv::Mat_<float> &cost)
{
    cv::Mat uint_dmb;
    cost.convertTo(uint_dmb,CV_8U,255.0/2.0);

    cv::imwrite("/home/xuan/MP-MVS/result/cost.jpg",uint_dmb);

    cv::namedWindow("dmap", (800,1200));
    cv::imshow("dmap",uint_dmb);
    cv::waitKey(0);   

}

