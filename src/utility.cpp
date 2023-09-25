#include "utility.h"

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
    fs["Geometric consistency planer prior"]>>config.geomPlanarPrior;
    fs["Sky segment"]>>config.sky_seg;
    //fs["depth map eval"]>>config.dmap_eval;
    //fs["depth map eval folder"]>>config.GT_folder;

    checkpath(config.input_folder);
    checkpath(config.output_folder);
    config.output_folder=config.output_folder+"/MPMVS";
    std::cout<<"Input data path:"<<config.input_folder<<std::endl;
    std::cout<<"Output data path:"<<config.output_folder<<std::endl;

    return config;
}
//read ETH3D GT depth map
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
    //cv::imwrite("/home/xuan/MP-MVS/result/GT.jpg",bgr_img);

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
    }
    closedir(pDir);

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
    //cv::imwrite("/home/xuan/MP-MVS/result/normal.jpg",normal);
    cv::imshow("normal",normal);
    cv::waitKey(0);
}


void GetHist(cv::Mat gray,cv::Mat &Hist)
{
    const int channels[1] = { 0 };
    float inRanges[2] = { 0,255 };
    const float* ranges[1] = {inRanges};
    const int bins[1] = { 256 };
    cv::calcHist(&gray, 1, channels,cv::Mat(), Hist,1, bins, ranges);
    //std::cout<<Hist<<std::endl;
    //ShowHist(Hist);
}


void ShowHist(cv::Mat &Hist)
{
    int hist_w = 512;
    int hist_h = 400;
    int width = 2;
    cv::Mat histImage = cv::Mat::zeros(hist_h,hist_w,CV_8UC3);
    for (int i = 0; i < Hist.rows; i++)
    {
        cv::rectangle(histImage,cv::Point(width*(i),hist_h-1),cv::Point(width*(i+1),hist_h-cvRound(Hist.at<float>(i)/20)),
        cv::Scalar(255,255,255),-1);
    }
    //cv::imwrite("/home/xuan/MP-MVS/result/hist.jpg",histImage);
    cv::namedWindow("histImage", cv::WINDOW_AUTOSIZE);
    cv::imshow("histImage", histImage);
    cv::waitKey(0);
}

int getMax10(cv::Mat Hist, int allnum){
    int num=0;
    for(int i=1;i<256;++i)
    {
        num+=(int)Hist.at<float>(i);
        if(((float)num/(float)allnum)>0.03)
            return i-1;
    }
    return 1;
}

int getMin10(cv::Mat Hist, int allnum){
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
    if(depth.empty())
    {   
        std::cout<<"Can not read this depth image !"<<std::endl;  
        exit(0);
    }
    double max, min=0;
    const int rows=depth.rows;
    const int cols=depth.cols;
    cv::Mat mask = cv::Mat::ones(depth.size(), CV_8UC1);
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            const float val=depth.at<float>(i,j);
            if(val<=0.0){
                mask.at<uchar>(i,j) = 0;
                depth.at<float>(i,j) = 0;
            }
            else{
                mask.at<uchar>(i,j) = 255;
            }
        }
    }
    cv::Point maxLoc;
    cv::Point minLoc;
    cv::minMaxLoc(depth, &min, &max, &minLoc, &maxLoc);
    std::cout<<"min:"<<min<<"  max:"<<max<<std::endl;

    double inv_max = 1 / (max-min+1e-8);
    cv::Mat norm_dmb= depth - min;
    norm_dmb = norm_dmb * inv_max;

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
    int top= getMax10(Hist, rows * cols);
    int down= getMin10(Hist, rows * cols);
    std::cout<<"top:"<<top<<std::endl;
    std::cout<<"down:"<<down<<std::endl;
    double new_min = min + (max - min)*((float)top / 256.0);
    double new_max = min + (max - min)*((float)down / 256.0);
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            const float val = depth.at<float>(i,j);
            if(val<new_min){
                depth.at<float>(i,j) = new_min;
                continue;
            }
            if(val>new_max)
                depth.at<float>(i,j) = new_max;
        }
    }
    inv_max = 1 / ( new_max - new_min + 1e-8);
    norm_dmb = depth - new_min + 1e-8;
    norm_dmb = norm_dmb * inv_max;
    norm_dmb.convertTo(uint_dmb,CV_8UC3,255.0);
    cv::Mat temp;
    cv::applyColorMap(uint_dmb,temp,cv::COLORMAP_JET);
    cv::Mat bgr_img2 = cv::Mat::zeros(rows,cols,CV_8UC3);
    Colormap2Bgr(temp,bgr_img2,mask);
    //cv::imwrite("/home/xuan/MP-MVS/result/"+name,bgr_img2);
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

