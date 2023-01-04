#include "inference.h"
#include "tools/decode.h"
#include <vector>
#include <string>
#define loop 10
#define Use_Simple_PostRun false
#define SAVE_PATH "../../align_bin"

cv::Mat my_draw_objects(cv::Mat& bgr, const std::vector<Phone::Object>& objects, double fps)
{
    static const char* class_names[] = {
            "phone"};
    cv::Mat image = bgr.clone();
    char text1[256];
    sprintf(text1, "fps=%.3f", fps);
    int baseline1 = 0;
    cv::Size label_size1 = cv::getTextSize(text1, cv::FONT_HERSHEY_SIMPLEX, 1, 2, &baseline1);
    cv::putText(image, text1, cv::Point(image.cols - label_size1.width, image.rows - label_size1.height), cv::FONT_HERSHEY_SIMPLEX, 1,cv::Scalar(0, 0, 0));

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Phone::Object& obj = objects[i];

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
    return image;
}

/* 此函数用来实时测试的demo
 * 读取默认的摄像头，同时送入推理接口进行检测，最后显示在detect窗口*/
int test_runtime_phone(){
    cv::VideoCapture capture(0);

    if (!capture.isOpened()) {
        std::cout << "open camera error!" << std::endl;
        return -1;
    }
    std::string model_param_file = "../../models/phonedet.enprm";
    std::string model_bin_file = "../../models/phonedet.enbin";
    Phone::InferRk model(model_param_file, model_bin_file);
    std::vector<Phone::Object> result;
    cv::Mat frame;
    while (1) {
        capture >> frame;
        if (frame.empty()) {
            std::cout << "capture empty frame" << std::endl;
            continue;
        }
        double start = get_current_time();
        result = model.Run(frame);
        double end = get_current_time();
        double fps = 1000.0f / (end - start);

        cv::Mat shrink_frame = my_draw_objects(frame, result, fps);
        result.clear();
        /*cv::resize(frame, shrink_frame,
                   cv::Size(frame.cols / 2, frame.rows / 2),
                   0, 0, 3);*/

        cv::imshow("detect", shrink_frame);

        int key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
    }
    capture.release();
    return 0;
}

int test_inference_phone(){

    //加密的文件模型
    std::string model_param_file = "../../models/phonedet.enprm";
    std::string model_bin_file = "../../models/phonedet.enbin";
    std::string img_file = "../../images/0002.jpg";
    std::string save_file = "../../results/yolox-phone.jpg";
    Phone::InferRk model(model_param_file, model_bin_file);
    printf("finish model\n");
    if(Use_Simple_PostRun)
    {
        //如果需要使用简化的后处理接口，则使用类方法simplify_postrun()进行设置
        model.simplify_postrun();
    }
    printf("begin read images\n");
    printf("%s\n", img_file.c_str());
    cv::Mat input = cv::imread(img_file.c_str());
    printf("finsh read images\n");
    std::vector<Phone::Object> result;
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for(size_t i = 0;i < loop; i++)
    {
        printf("%d------------\n", i);
        double start = get_current_time();
        //推理接口，传入cv::Mat即可返回result结果，同rk3399-tengine的几乎一致
        cv::Mat now1 = input.clone();
        result = model.Run(now1);
        size_t result_num = result.size();
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        if (min_time > cur)
            min_time = cur;
        if (max_time < cur)
            max_time = cur;
        //model.align_tool(SAVE_PATH);
        if(i < loop - 1) {
            result.clear();
        }
    }
    printf("Result Size Numbers is %d\n", result.size());
    printf("Repeat %d times, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", loop,
           total_time / loop, max_time, min_time);
    if(Use_Simple_PostRun)
    {
        if(result.size() > 0)
        {
            printf("There are some phone target in photo\n");
        }
    }
    else
    {
        //画框并且保存图片
        model.draw_object(input, result, save_file);
    }
    return 0;
}

int show_usage(){
    printf("Press any key to continue during the detection process, and press q to exit if you want to quit\n");
    printf("Press any key to start\n");
    getchar();
    return 0;
}

int main (int argc, char** argv){

    if (argc >= 2){
        std::string option = std::string(argv[1]);
        if (option == "help") {
            show_usage();
        }else if (option == "camera"){
            return test_runtime_phone();
        }else if (option == "inference_time"){
            return test_inference_phone();
        }else{
            return test_inference_phone();
        }
    }
    else{
        show_usage();
        test_inference_phone();
    }
    return 0;
}

