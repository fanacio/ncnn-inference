
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <tools/decode.h>
#include "inference.h"
#include "net.h"
#include "layer.h"


//以下的宏跟前处理相关
#define DEFAULT_IMG_H        416
#define DEFAULT_IMG_W        416
#define DEFAULT_IMG_C        3
#define DEFAULT_SCALE1       (0.229f / 255)
#define DEFAULT_SCALE2       (0.224f / 255)
#define DEFAULT_SCALE3       (0.225f / 255)
#define DEFAULT_MEAN1        (0.485 * 255)
#define DEFAULT_MEAN2        (0.456 * 255)
#define DEFAULT_MEAN3        (0.406 * 255)
//以下的宏跟tengine相关
#define ACLM                 false
#define PERMUTE_UESFULL      true
#define DEFAULT_THREAD_COUNT 1
//下面的宏用于后处理
#define OUTPUT_N             1
#define OUTPUT_C             3549
#define OUTPUT_HW            6
#define GRIDS_HW             2
#define STRIDES_HW           1
#define NUM_CLASSES          1
#define CONF_THRE            0.25
#define DETECTIONS_DIMS      7
#define CLASS_AGNOSTIC       true
#define NMS_THRE             0.45
#define LEGACY               false

#define Simplify_Postrun     false
// 分别是lib/common内的bin路径，可根据需要进行修改
#define GRIDS_BIN_PATH       "../../lib/common/grids.bin"
#define STRIDES_BIN_PATH     "../../lib/common/strides.bin"

namespace Phone{
    /* 用于存放读取文件的结构体
     * */
    struct MyFile{
        size_t size;
        void* ptr;
    };
    // 注册算子进去
    class Upsample : public ncnn::Layer{
    public:
        Upsample(){
            one_blob_only = true;
        }
        virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;

            int outw = w * 2;
            int outh = h * 2;
            int outc = channels;

            top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outc; p++)
            {
                const float* ptr = bottom_blob.channel(p);
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    const float* ptr_h = ptr + i / 2 * w;
                    float* outptr_h = outptr + i * outw;
                    for (int j = 0; j < outw; j++)
                    {
                        *(outptr_h + j) = *(ptr_h + j / 2);
                    }
                }
            }

            return 0;
        }
    };
    DEFINE_LAYER_CREATOR(Upsample)
    // 确定文件是否存在
    bool check_file_exist(const char* file_name)
    {
        FILE* fp = fopen(file_name, "r");
        if (!fp)
        {
            //fprintf(stderr, "Input file not existed: %s\n", file_name);
            return false;
        }
        fclose(fp);
        return true;
    }
    int getBinSize(std::string path, bool info=false)
    {
        int size = 0;
        std::ifstream infile(path, std::ifstream::binary);

        infile.seekg(0, infile.end);
        size = infile.tellg();
        infile.seekg(0,infile.beg);

        infile.close();
        if(info){
            fprintf(stderr, "\npath=%s, size=%d \n",path.c_str() , size);
        }
        return size;
    }

    void readBin(std::string path, char *buf , int size)
    {
        std::ifstream infile(path, std::ifstream::binary);

        infile.read(static_cast<char *>(buf), size);
        infile.close();
    }

    void writeBin(std::string path, char* buf, int size){
        std::ofstream outfile(path, std::ifstream::binary);
        outfile.write((char*)(buf), size);
        outfile.close();
    }

    float* grids_strides(std::string filePath)
    {
        int size = getBinSize(filePath);
        char *buf = new char [size];
        readBin(filePath, buf, size);
        float *fbuf = reinterpret_cast<float *>(buf);
        return fbuf;
    }
    /* 读取文件，同时返回MyFile结构体
     * */
    struct MyFile readMyFile(std::string filePath){
        struct MyFile onefile;
        onefile.size = getBinSize(filePath);
        onefile.ptr = new char[onefile.size];
        readBin(filePath, (char*)onefile.ptr, onefile.size);
        return onefile;
    }
    float intersection_area(const Object& a, const Object& b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }
    //nms 算法
    void nms_sorted_bboxes(const std::vector<Object>& yoloxobjects, std::vector<int>& picked, float nms_threshold)
    {
        picked.clear();
        
        const int n = yoloxobjects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++)
        {
            areas[i] = yoloxobjects[i].rect.area();
        }

        for (int i = 0; i < n; i++)
        {
            const Object& a = yoloxobjects[i];

            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++)
            {
                const Object& b = yoloxobjects[picked[j]];

                // intersection over union
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                // float IoU = inter_area / union_area
                if (inter_area / union_area > nms_threshold)
                {
                    picked[j] = a.prob > b.prob ? i : picked[j];
                    keep = 0;
                }
            }
            if (keep)
                picked.push_back(i);
        }
    }
    //画框接口
    void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::string save_path)
    {
        static const char* class_names[] = {
            "phone"};

        cv::Mat image = bgr.clone();

        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];

            fprintf(stderr, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
                    obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

            cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1, 2, &baseLine);

            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image.cols)
                x = image.cols - label_size.width;

            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                        cv::Scalar(255, 255, 255), -1);

            cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(0, 0, 0));
        }

        cv::imwrite(save_path, image);
    }
    // 最重要的推理接口
    class Infer{
    private:
//        bool _cfg_shape();
        void _process_input_ncnn(const cv::Mat& org_input);
//        void _process_input(const cv::Mat org_img);
        void _inference();
        std::vector<Object> _post_run();
        std::vector<Object> _post_run_simplify();
        int _img_h = DEFAULT_IMG_H;
        int _img_w = DEFAULT_IMG_W;
        int _img_c = DEFAULT_IMG_C;
        int _batchsize = OUTPUT_N;
        int _out_c = OUTPUT_C;
        int _out_hw = OUTPUT_HW;
        int _grids_hw = GRIDS_HW;
        int _strides_hw = STRIDES_HW;
        int _num_classes = NUM_CLASSES;
        float _conf_thre = CONF_THRE;
        int _detections_dims = DETECTIONS_DIMS;
        bool _class_agnostic = CLASS_AGNOSTIC;
        float _nms_thre = NMS_THRE;
        bool _legacy = LEGACY;
        bool _simplify_postrun = Simplify_Postrun;
        float _ratio = 1.0f;
        const float _mean[3] = {DEFAULT_MEAN1, DEFAULT_MEAN2, DEFAULT_MEAN3};
        const float _scale[3] = {DEFAULT_SCALE1, DEFAULT_SCALE2, DEFAULT_SCALE3};
        //原始的图像的尺寸信息
        int _raw_w = 0;
        int _raw_h = 0;
        /* 用于进行加密和解密的运行时变量 */
        void* param_mem = nullptr;
        void* bin_mem = nullptr;
        /* 两个bin加载到内存中的变量 */
        float* _grids = nullptr;
        float* _strides = nullptr;
        /* ncnn推理引擎相关的变量 */
        ncnn::Net yolox;
//        ncnn::Extractor yolox_ex;
        ncnn::Mat in;
        ncnn::Mat padding_in;
        ncnn::Mat out;
        ncnn::Mat cat_blob14;
        ncnn::Mat cat_blob15;
        ncnn::Mat cat_blob16;
        /*为了后处理而预先设置的内存空间*/
        float* _class_conf = nullptr;
        int* _class_pred = nullptr;
        //bool* _conf_mask = nullptr;
        float* _detections = nullptr;
        float* _o_detections = nullptr;
        // float* _input_data = nullptr;
        float* _output_data = nullptr;
        // permute函数
        void _permute();
        int _o_dims[4] = {0};
    public:
        //默认的初始化函数，使用后无效
        Infer(){};
        /* 初始化函数
         * 默认是加密文件传入其中
         * */
        Infer(std::string model_param_path, std::string model_bin_path);
        ~Infer();
        /* 默认是加密文件传入其中，所以encrypt=true
         * */
        void InferRe(std::string model_param_path, std::string model_bin_path, bool encrypt=true);

        // 推理函数
        std::vector<struct Object> Run(cv::Mat& org_img);
        // 后处理参数重配置函数
        void cfg_postrun(float nms_thre = NMS_THRE, float conf_thre = CONF_THRE);
        void simplify_postrun();
        //用于对齐的工具
//        void save_tensors(std::string save_path);
//        void align_tool(std::string org_save_path);
    };

    Infer::Infer(std::string model_param_path, std::string model_bin_path)
    {
        InferRe(model_param_path, model_bin_path);
    }
    

    void Infer::InferRe(std::string model_param_path, std::string model_bin_path, bool encrypt){
        /*ncnn加载*/
        this->yolox.register_custom_layer("Upsample", Upsample_layer_creator);
        if(encrypt){
            /*加密进行读取
             * */
//            struct MyFile onefile = readMyFile(model_param_path);
//            struct MyFile binfile = readMyFile(model_bin_path);
            /* 使用加密算法进行文件解密，从而剥离出二进制流
             * */
            struct MyFile onefile;
            struct MyFile binfile;
            printf("waitting for init\n");
            decryptFileAndReture(model_param_path, &(onefile.ptr), &(onefile.size));
            decryptFileAndReture(model_bin_path, &(binfile.ptr), &(binfile.size));
            this->yolox.load_param_encrypt(onefile.ptr, onefile.size);
//            printf("finish read param file\n");
            this->yolox.load_model_encrtpt(binfile.ptr, binfile.size);
            printf("finish init\n");
//            printf("finish read bin file\n");
            delete [] onefile.ptr;
            delete [] binfile.ptr;
        }else{
            /*不加密直接读取model文件
             * */
            this->yolox.load_param(model_param_path.c_str());
            this->yolox.load_model(model_bin_path.c_str());
        }
        /* 加载后处理的必要数据 */
        this->_grids = grids_strides(GRIDS_BIN_PATH);
        this->_strides = grids_strides(STRIDES_BIN_PATH);
        /*预先设置后处理的内存空间*/
        this->_class_conf = new float[this->_batchsize * this->_out_c];
        this->_class_pred = new int[this->_batchsize * this->_out_c];
        //this->_conf_mask = new bool[this->_batchsize * this->_out_c];
        this->_detections = new float[this->_detections_dims * this->_out_c];
        this->_o_detections = new float[this->_detections_dims * this->_out_c];
        this->_output_data = new float[this->_out_c * this->_out_hw];
    }
    Infer::~Infer(){
        delete [] this->_grids;
        delete [] this->_strides;
        delete [] this->_class_conf;
        delete [] this->_class_pred;
        delete [] this->_detections;
        delete [] this->_o_detections;
        delete [] this->_output_data;
    }


    // 从模型结构体得到输入输出的形状信息，同时进行设置
    // 目前默认是NCHW排布
//    bool Infer::_cfg_shape(){
//        int dims[4];
//        // 配置输入形状
//        if(get_tensor_shape(this->_input_tensor, dims, 4) != 0)
//        {
//            fprintf(stderr, "get input shape failed\n");
//            return false;
//        }
//        this->_batchsize = dims[0];
//        this->_img_c = dims[1];
//        this->_img_h = dims[2];
//        this->_img_w = dims[3];
//        // 配置输出和后处理信息
//        if (get_tensor_shape(this->_output_tensor, dims, 4) != 0)
//        {
//            fprintf(stderr, "get output shape failed\n");
//            return false;
//        }
//        this->_out_c = dims[1];
//        this->_out_hw = dims[2] * dims[3];
//        this->_num_classes = this->_out_hw - 5;
//        assert(this->_num_classes > 0);
//        return true;
//    }
    // permute函数，暂时使用
    void Infer::_permute()
    {
        
        int n1 = this->_o_dims[0];
        int c1 = this->_o_dims[1];
        int h1 = this->_o_dims[2];
        int w1 = this->_o_dims[3];
        //float* org_output = (float*)get_tensor_buffer(this->_output_tensor);
        float* org_output = nullptr;
        fprintf(stderr, "%d, %d, %d, %d\n", n1, c1, h1, w1);
        // nchw -> nhcw == 1 * 
        for (size_t i = 0; i < n1; i++)
        {
            for (size_t j = 0; j < c1; j++)
            {
                for (size_t z = 0; z < h1; z++)
                {
                    for (size_t t = 0; t < w1; t++)
                    {
                        this->_output_data[i*h1*c1*w1+z*c1*w1+j*w1+t] = org_output[i*c1*h1*w1 + j*h1*w1+z*w1+t];
                    }
                }
            }
        }
    }
    // 推理时函数，整个函数
    std::vector<struct Object> Infer::Run(cv::Mat& org_img){
        fprintf(stderr, "process  begin\n"); 
        _process_input_ncnn(org_img);
        fprintf(stderr, "infer  begin\n"); 
        _inference();
        fprintf(stderr, "post  begin\n");
        if (this->_simplify_postrun)
        {
            return _post_run_simplify();
        }
        else
        {
            return _post_run();
        }
        //return _post_run_simplify();
        //std::vector<struct Object> a;
        //return a;
    }
    // 后处理参数重新配置函数
    void Infer::cfg_postrun(float nms_thre, float conf_thre)
    {
        this->_nms_thre = nms_thre;
        this->_conf_thre = conf_thre;
    }
    // 配置是否阉割后处理运行
    void Infer::simplify_postrun()
    {
        this->_simplify_postrun = true;
    }
    // 前处理函数
//    void Infer::_process_input(cv::Mat org_mat){
//        int raw_w = org_mat.cols;
//        int raw_h = org_mat.rows;
//        this->_raw_w = raw_w;
//        this->_raw_h = raw_h;
//        this->_ratio = std::min((float)this->_img_w / raw_w, (float)this->_img_h / raw_h);
//        cv::Mat sample;
//        int now_w = std::min(this->_img_w, (int)std::ceil(raw_w * this->_ratio));
//        int now_h = std::min(this->_img_h, (int)std::ceil(raw_h * this->_ratio));
//        cv::resize(org_mat, sample, cv::Size(now_w, now_h));
//        sample.convertTo(sample, CV_32FC3);
//        //Fill size
//        cv::copyMakeBorder(sample, sample, 0, this->_img_h - now_h, 0,
//                            this->_img_w - now_w, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
//	// fprintf(stderr, "sample size is w-%d, h-%d\n", sample.cols, sample.rows);
//        float* img_data = (float*)sample.data;
//        float* input_buffer = (float*) this->_input_buffer;
//
//        /* nhwc to nchw */
//        if (this->_aclm == false)
//        {
//            if (this->_legacy)
//            {
//
//                for (int h = 0; h < this->_img_h; h++)
//                {
//                    for (int w = 0; w < this->_img_w; w++)
//                    {
//                        for (int c = 0; c < 3; c++)
//                        {
//                            int in_index = h * this->_img_w * 3 + w * 3 + c;
//                            int out_index = c * this->_img_h * this->_img_w + h * this->_img_w + w;
//                            input_buffer[out_index] = (img_data[in_index] - this->_mean[c]) * this->_scale[c];
//                        }
//                    }
//                }
//            }
//            else
//            {
//                for (int h = 0; h < this->_img_h; h++)
//                {
//                    for (int w = 0; w < this->_img_w; w++)
//                    {
//                        for (int c = 0; c < 3; c++)
//                        {
//                            int in_index = h * this->_img_w * 3 + w * 3 + c;
//                            int out_index = c * this->_img_h * this->_img_w + h * this->_img_w + w;
//                            input_buffer[out_index] = (img_data[in_index]) ;
//                        }
//                    }
//                }
//            }
//        }
//        else
//        {
//            memcpy(input_buffer, img_data, sizeof(float) * this->_img_h * this->_img_w * this->_img_c);
//        }
//    }
    //利用ncnn的前处理接口
    void Infer::_process_input_ncnn(const cv::Mat& org_mat){
        int raw_w = org_mat.cols;
        int raw_h = org_mat.rows;
        this->_raw_w = raw_w;
        this->_raw_h = raw_h;
        this->_ratio = std::min((float)this->_img_w / raw_w, (float)this->_img_h / raw_h);
        int now_w = std::min(this->_img_w, (int)std::ceil(raw_w * this->_ratio));
        int now_h = std::min(this->_img_h, (int)std::ceil(raw_h * this->_ratio));
        this->in = ncnn::Mat::from_pixels_resize(org_mat.data, ncnn::Mat::PIXEL_BGR, raw_w, raw_h, now_w, now_h);
        ncnn::copy_make_border(this->in, this->padding_in, 0, this->_img_h - now_h, 0,
                               this->_img_w - now_w, ncnn::BORDER_CONSTANT, 114.f);
    }
//    void mmmm(){
//        ncnn::Net yolox;
//        yolox.load_param("yolox.param");
//        yolox.load_model("yolox.param.bin");
//        //1.生成输入变量input
//        cv::Mat input_mat = cv::imread("img.jpg");
//        ncnn::Mat input = ncnn::Mat::from_pixels_resize(input_mat.data, ncnn::Mat::PIXEL_BGR, input_mat.cols, input_mat.rows, 256,256);
//        //2.进行推理
//        ncnn::Extractor yolox_ex = yolox.create_extractor();
//        yolox_ex.input("input_blob", input);
//        ncnn::Mat output;
//        yolox_ex.extract("output_blob",output);
//        //3.打印或者使用output的内部信息
//        for (int q=0; q<output.c; q++)
//        {
//            const float* ptr = output.channel(q);
//            for (int y=0; y<output.h; y++)
//            {
//                for (int x=0; x<output.w; x++)
//                {
//                    printf("%f ", ptr[x]);
//                }
//                ptr += output.w;
//                printf("\n");
//            }
//            printf("------------------------\n");
//        }
//    }
    void print_mat(ncnn::Mat& a, std::string name){
        printf("<-----------------%s------------->\n", name.c_str());
        printf("dims number is %d\n", a.dims);
        printf("dims is c=%d, d=%d, h=%d, w=%d\n", a.c, a.d, a.h, a.w);
    }
    // 推理函数
    void Infer::_inference(){
        ncnn::Extractor yolox_ex = this->yolox.create_extractor();
        yolox_ex.input("data", this->padding_in);
//        ncnn::Mat upsample;
//        ncnn::Mat flatten_blob1;

//        yolox_ex.extract("relu_blob55_splitncnn_1", upsample);
        yolox_ex.extract("cat_blob14",cat_blob14);
        yolox_ex.extract("cat_blob15",cat_blob15);
        yolox_ex.extract("cat_blob16",cat_blob16);
//        yolox_ex.extract("cat_blob17", this->out);
        //yolox_ex.extract("flatten_blob1", flatten_blob1);
//        print_mat(upsample, "upsample input");
//        print_mat(cat_blob14, "cat_blob14");
//        print_mat(cat_blob15, "cat_blob15");
//        print_mat(cat_blob16, "cat_blob16");
//        print_mat(this->out, "cat_blob17");
        //print_mat(flatten_blob1, "flatten_blob1");
        // 调试代码
//        ncnn::Mat concat_blob;
//        yolox_ex.extract("cat_blob17", concat_blob);
//        printf("permute out dims is %d, %d, %d, %d\n", concat_blob.w, concat_blob.h, concat_blob.d, concat_blob.c);
//        printf("concat_blob dims number is %d\n", this->out.dims);
//        printf("permute out dims is %d, %d, %d, %d\n", this->out.w, this->out.h, this->out.d, this->out.c);
//        memcpy(this->_output_data, this->out.channel(0), sizeof(float)*this->_out_c*this->_out_hw);
        /* 利用三个cat的输入来组合成最终的结果 */
        for(size_t i = 0; i < this->_out_hw; i++){
            size_t address_step = 0;
            //代替flatten+concat+permute操作
            const float* o_big_52 = cat_blob14.channel(i);
            const float* o_mid_26 = cat_blob15.channel(i);
            const float* o_min_13 = cat_blob16.channel(i);
            size_t big_size = cat_blob14.h * cat_blob14.w;
            size_t mid_size = cat_blob15.h * cat_blob15.w;
            size_t min_size = cat_blob16.h * cat_blob16.w;
            for(size_t j = 0; j < big_size; j++){
                this->_output_data[address_step * this->_out_hw + i] = o_big_52[j];
                address_step++;
            }
            for(size_t j = 0; j < mid_size; j++){
                this->_output_data[address_step * this->_out_hw + i] = o_mid_26[j];
                address_step++;
            }
            for(size_t j = 0; j < min_size; j++){
                this->_output_data[address_step * this->_out_hw + i] = o_min_13[j];
                address_step++;
            }
        }

        //this->_output_data = this->out.channel(0);
    }
    // 后处理函数
    std::vector<Object> Infer::_post_run(){
        printf("postrun step1\n");
        // 类似于yolo层
        for (size_t i = 0; i < this->_out_c; i++)
        {
//            printf("index-%d\n", i);
            this->_output_data[this->_out_hw*i] = (this->_output_data[this->_out_hw*i]+this->_grids[this->_grids_hw*i])*this->_strides[i];
            this->_output_data[this->_out_hw*i+1] = (this->_output_data[this->_out_hw*i+1]+this->_grids[this->_grids_hw*i+1])*this->_strides[i];
            this->_output_data[this->_out_hw*i+2] = std::exp(this->_output_data[this->_out_hw*i+2])*this->_strides[i];
            this->_output_data[this->_out_hw*i+3] = std::exp(this->_output_data[this->_out_hw*i+3])*this->_strides[i];
        }
        printf("postrun step2\n");
        // 获取最大类别得分对应的序号和背景得分
        for (size_t i = 0; i < this->_out_c; i++)
        {
            float maxconf = 0.f ;
            int max_pred = 0 ;
            for (size_t j = 5; j < 5 + this->_num_classes; j++)
            {
                if (this->_output_data[this->_out_hw*i+j] > maxconf)
                {
                    maxconf = this->_output_data[this->_out_hw*i+j];
                    max_pred = j - 5;
                }
            }
            this->_class_conf[i] = maxconf;
            this->_class_pred[i] = max_pred;
        }
        printf("postrun step3\n");
        // 获得大于阈值的框的序号
        std::vector<int> index ;
        for (size_t i = 0; i < this->_out_c; i++)
        {
            if (this->_output_data[this->_out_hw*i+4] * (float)this->_class_conf[i] >= this->_conf_thre)
            {
                index.push_back(i);
            }
        }
        int count = index.size();
        printf("postrun step4\n");
        // 解码输出，得到7维的框结果:(x1,y1,x2,y2,obj_conf,class_conf,class_pred)
        for (size_t i = 0; i < this->_out_c; i++)
        {
            for (size_t j = 0; j < 5; j++)
            {
                this->_detections[this->_detections_dims * i+j] = this->_output_data[this->_out_hw * i + j];
            }
            this->_detections[this->_detections_dims * i+5] = this->_class_conf[i];
            this->_detections[this->_detections_dims * i+6] = this->_class_pred[i];
        }
        for (size_t i = 0; i < count; i++)
        {
            for (size_t j = 0; j < this->_detections_dims; j++)
            {
                this->_o_detections[this->_detections_dims * i + j] = this->_detections[this->_detections_dims * index[i] + j];
            }    
        }
        printf("postrun step5\n");
        // nms，以及得出最终的结果
        std::vector<Object> o_objects;
        if (count != 0)
        {
            if (this->_class_agnostic)
            {
                std::vector<int> picked;
                std::vector<Object> objects;
                for (size_t i = 0; i < count; i++)
                {
                    Object obj;
                    obj.rect.x = this->_o_detections[this->_detections_dims* i]- this->_o_detections[this->_detections_dims * i + 2]/2;
                    obj.rect.y = this->_o_detections[this->_detections_dims * i + 1]- this->_o_detections[this->_detections_dims * i + 3]/2;
                    obj.rect.width = this->_o_detections[this->_detections_dims * i + 2];
                    obj.rect.height = this->_o_detections[this->_detections_dims * i + 3];
                    obj.label = (int)this->_o_detections[this->_detections_dims * i + 6];
                    obj.prob = this->_o_detections[this->_detections_dims * i + 4] * this->_o_detections[this->_detections_dims * i + 5];
                    objects.push_back(obj);
                }
                /* NMS */
                nms_sorted_bboxes(objects, picked, this->_nms_thre);

                int objects_num = picked.size();
                o_objects.resize(objects_num);
                for (int i = 0; i < objects_num; i++)
                {
                    o_objects[i] = objects[picked[i]];
                    float x1 = (o_objects[i].rect.x);
                    float y1 = (o_objects[i].rect.y);
                    float x2 = (o_objects[i].rect.x + o_objects[i].rect.width);
                    float y2 = (o_objects[i].rect.y + o_objects[i].rect.height);

                    x1 = x1 / this->_ratio;
                    y1 = y1 / this->_ratio;
                    x2 = x2 / this->_ratio;
                    y2 = y2 / this->_ratio;

                    x1 = std::max(std::min(x1, (float)(this->_raw_w - 1)), 0.f);
                    y1 = std::max(std::min(y1, (float)(this->_raw_h - 1)), 0.f);
                    x2 = std::max(std::min(x2, (float)(this->_raw_w - 1)), 0.f);
                    y2 = std::max(std::min(y2, (float)(this->_raw_h - 1)), 0.f);

                    o_objects[i].rect.x = x1;
                    o_objects[i].rect.y = y1;
                    o_objects[i].rect.width = x2 - x1;
                    o_objects[i].rect.height = y2 - y1;
                }
            }
            else
            {
                fprintf(stderr, "This feature needs to be improved.\n");
            }
        }
        return o_objects;
    }

    //阉割版后处理，用于落地使用
    std::vector<Object> Infer::_post_run_simplify(){
        // 获取最大类别得分对应的序号和背景得分
        std::vector<Object> o_objects;
        for (size_t i = 0; i < this->_out_c; i++)
        {
            float maxconf = 0.f ;
            for (size_t j = 5; j < 5 + this->_num_classes; j++)
            {
                if (this->_output_data[this->_out_hw*i+j] > maxconf)
                {
                    maxconf = this->_output_data[this->_out_hw*i+j];
                }
            }
            this->_class_conf[i] = maxconf;
            if (this->_output_data[this->_out_hw*i+4] * (float)this->_class_conf[i] >= this->_conf_thre)
            {
                o_objects.resize(1);
                return o_objects;
            }
        }
        return o_objects;
    }

    bool if_in_vector(std::vector<std::string> listt, std::string a){
        return (std::find(listt.begin(), listt.end(), a) == listt.end())?false:true;
    }
    // 距离计算
    double L1Distance(float* a, float* b, size_t num){
        double loss = std::abs(a[0] - b[0]) / std::max((std::abs(a[0]) + std::abs(b[0]))/2.0f, 0.000001f);
        for (size_t i = 1; i < num ; i++){
            loss = loss  + (std::abs(a[i] - b[i])/std::max((std::abs(a[i]) + std::abs(b[i]))/2.0f, 0.000001f) - loss)/(i+1);
        }
        return loss;
    }
    Infer inferi;

    InferRk::InferRk(std::string model_param_path, std::string model_bin_path)
    {
        inferi.InferRe(model_param_path, model_bin_path);
    }

    std::vector<Object> InferRk::Run(cv::Mat& img)
    {
        return inferi.Run(img);
    }

    void InferRk::draw_object(const cv::Mat& bgr, const std::vector<Object>& objects, std::string save_path)
    {
        draw_objects(bgr, objects, save_path);
    }

    void InferRk::cfg_postrun(float nms_thre, float conf_thre)
    {
        inferi.cfg_postrun(nms_thre, conf_thre);
    }
    void InferRk::simplify_postrun()
    {
        inferi.simplify_postrun();
    }
//    void InferRk::save_tensors(std::string save_path){
//        inferi.save_tensors(save_path);
//    }
//    void InferRk::align_tool(std::string save_path){
//        inferi.align_tool(save_path);
//    }

}
