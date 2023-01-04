#ifndef _PHONE_CPU_NCNN_H_
#define _PHONE_CPU_NCNN_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common.h"

#ifndef _MSC_VER
#define _MSC_VER 1929
#endif
#define OBJECT_DETECTOR_EXPORTS
#ifdef _WIN32
#ifdef  OBJECT_DETECTOR_EXPORTS
#define  OBJ_DET_API  __declspec(dllexport)
#else
#define  OBJ_DET_API  __declspec(dllimport)
#endif  // !OBJECT_DETECTOR_EXPORTS
#else
#define OBJ_DET_API
#endif
namespace Phone
{
    struct Object
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
    };

    class OBJ_DET_API InferRk{
    private:
        
    public:
        //default is used
        InferRk(){};

        // this is the right build function
        InferRk(std::string model_param_path, std::string model_bin_path);

        ~InferRk(){};

        // the func as inference
        std::vector<struct Object> Run(cv::Mat& org_img);

        //the func as draw object
        void draw_object(const cv::Mat& bgr, const std::vector<Object>& objects, std::string save_path);

        // configure the postrun info for detect
        void cfg_postrun(float nms_thre, float conf_thre);
        
        // configure the postrun to determine if simplify_postrun
        void simplify_postrun();

        //align_tools
//        void save_tensors(std::string save_path);
//        void align_tool(std::string save_path);
    };

} // namespace Phone
#endif // !_PHONE_CPU_NCNN_H_