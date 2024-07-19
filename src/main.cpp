#include "meter_seg.h"
#include "meter_reader.h"
#include "meter_detect.h"
#include "nanodet.h"

#include <chrono>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define DET_PARAM  "/home/hit/Project/MeterReader_Image/weights/nanodet.param"
#define DET_BIN "/home/hit/Project/MeterReader_Image/weights/nanodet.bin"
#define SEG_PARAM "/home/hit/Project/MeterReader_Image/weights/fastseg.param"
#define SEG_BIN "/home/hit/Project/MeterReader_Image/weights/fastseg.bin"
#define SAVE_DIR "/home/hit/Project/MeterReader_Image/outputs/"

std::unique_ptr<MeterDetection> meterDet(new MeterDetection(DET_PARAM, DET_BIN));
std::unique_ptr<MeterSegmentation> meterSeg(new MeterSegmentation(SEG_PARAM, SEG_BIN));
std::unique_ptr<NanoDet> nanoDet(new NanoDet(DET_PARAM, DET_BIN, false));

MeterReader meterReader;
// MeterReaderV2 meterReaderV2;
// std::vector<READ_RESULT> read_results;

#define SAVE_IMG

void detectProcess(const cv::Mat& input_image, int index) {
    if (input_image.empty()) {
        std::cerr << "cv::imread read image failed" << std::endl;
        return;
    }

    std::vector<Object> objects;
    object_rect effect_roi;
    cv::Mat resized_img;
    nanoDet->resize_uniform(input_image, resized_img, cv::Size(320, 320), effect_roi);
    auto results = nanoDet->detect(resized_img, 0.4, 0.3);
    nanoDet->convert_boxes(results, effect_roi, input_image.cols, input_image.rows, objects);

    std::cout << "Object size: " << objects.size() << std::endl;

    if (!objects.empty()) {
        std::vector<cv::Mat> meters_image = meterSeg->cut_roi_img(input_image, objects);

        std::vector<cv::Mat> meters_image_pad;
        meters_image_pad.clear();
        std::vector<cv::Mat> seg_result = meterSeg->preprocess(meters_image, meters_image_pad);

        std::vector<float> scale_values = meterReader.multi_read_process(meters_image_pad, seg_result);

#ifdef SAVE_IMG
        // 如果需要保存每个处理后的图像，可以将cv::imwrite放在循环内部，并修改文件名以避免覆盖
        cv::Mat save_frame = meterReader.result_visualizer(input_image, objects, scale_values);
        
        // 保存处理后的图像
        std::string save_path = std::string(SAVE_DIR) + std::to_string(index) + "_processed_image.jpg";
        cv::imwrite(save_path, save_frame);
        std::cout << "Saved processed image to " << save_path << std::endl;
#else
        for (const auto& scale_value:scale_values)
        {
            std::cout << "scale_value: " << meter_reader.floatToString(scale_value) + " Mpa" << std::endl;
        }
#endif
    } else {
        std::cout << "No objects detected." << std::endl;
    }
}

std::vector<cv::Mat> ReadImages(const std::string& pattern) 
{
    std::vector<cv::String> fn;
    cv::glob(pattern, fn, false);
    std::vector<cv::Mat> images;
    int count = fn.size(); // number of files in images folder
    for (int i = 0; i < count; i++) {
        images.emplace_back(cv::imread(fn[i]));
    }
    return images;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mode> <path>" << std::endl;
        std::cerr << "Mode: single - Process a single image" << std::endl;
        std::cerr << "      folder - Process all images in a folder" << std::endl;
        return -1;
    }

    std::string mode = argv[1];
    std::string path = argv[2];

    int index = 0;

    if (mode == "single") {
        cv::Mat image = cv::imread(path);

        if (image.empty()) {
            std::cerr << "Could not open or find the image at " << path << std::endl;
            return -1;
        }

        // 记录开始时间
        auto start = std::chrono::high_resolution_clock::now();

        // 调用 detectProcess 函数
        detectProcess(image, index);

        // 记录结束时间
        auto end = std::chrono::high_resolution_clock::now();

        // 计算持续时间
        std::chrono::duration<double, std::milli> elapsed = end - start;

        // 输出处理时间
        std::cout << "Processing time: " << elapsed.count() << " ms" << std::endl;

    } else if (mode == "folder") {
        std::vector<cv::Mat> images = ReadImages(path + "/*.jpg");

        if (images.empty()) {
            std::cerr << "No images found in the folder " << path << std::endl;
            return -1;
        }

        for (const auto& image : images) {
            // 记录开始时间
            auto start = std::chrono::high_resolution_clock::now();

            // 调用 detectProcess 函数
            detectProcess(image, index);

            // 记录结束时间
            auto end = std::chrono::high_resolution_clock::now();

            // 计算持续时间
            std::chrono::duration<double, std::milli> elapsed = end - start;

            // 输出处理时间
            std::cout << "Processing time: " << elapsed.count() << " ms" << std::endl;

            index++;
        }
    } else {
        std::cerr << "Invalid mode. Use 'single' or 'folder'." << std::endl;
        return -1;
    }

    std::cout << "Processing complete" << std::endl;
    return 0;
}

