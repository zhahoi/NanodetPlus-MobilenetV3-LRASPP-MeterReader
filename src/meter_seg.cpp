#include "meter_seg.h"
#include "gpu.h"
#include <mutex>
#include <iostream>
#include <thread>
#include <vector>

MeterSegmentation::MeterSegmentation(const char* param, const char* bin)
{
    meterSeg.opt.use_vulkan_compute = true;  // 启用Vulkan加速
    meterSeg.opt.use_bf16_storage = false;
    meterSeg.load_param(param);
    meterSeg.load_model(bin);
}

bool MeterSegmentation::run(const cv::Mat& img, ncnn::Mat& res)
{
    if (img.empty())
        return false;

    // 预处理
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, DEEPLABV3P_TARGET_SIZE, DEEPLABV3P_TARGET_SIZE);
    input.substract_mean_normalize(mean, std);

    ncnn::Extractor ex = meterSeg.create_extractor();
    ex.input("images", input);
    ex.extract("output", res);

    // Softmax
    Softmax(res);

    return true;
}

MeterSegmentation::~MeterSegmentation()
{
    meterSeg.clear();
}

float MeterSegmentation::ResizeImage(const cv::Mat& image, cv::Mat& out_image, const cv::Size& new_shape, const cv::Scalar& color = cv::Scalar(128, 128, 128)) {
    cv::Size shape = image.size();
    float scale = std::min(static_cast<float>(new_shape.height) / shape.height, static_cast<float>(new_shape.width) / shape.width);

    int new_width = static_cast<int>(shape.width * scale);
    int new_height = static_cast<int>(shape.height * scale);

    cv::resize(image, out_image, cv::Size(new_width, new_height), 0, 0, cv::INTER_AREA);

    if (new_width < new_shape.width || new_height < new_shape.height) {
        cv::copyMakeBorder(out_image, out_image, 
            (new_shape.height - new_height) / 2, (new_shape.height - new_height + 1) / 2,
            (new_shape.width - new_width) / 2, (new_shape.width - new_width + 1) / 2,
            cv::BORDER_CONSTANT, color);
    }

    return scale;
}

void MeterSegmentation::Softmax(ncnn::Mat& res)
{
    auto softmax_channel = [&](int c) {
        for (int i = 0; i < res.h; i++) {
            for (int j = 0; j < res.w; j++) {
                float max = -FLT_MAX;
                for (int q = 0; q < res.c; q++) {
                    max = std::max(max, res.channel(q).row(i)[j]);
                }

                float sum = 0.0f;
                for (int q = 0; q < res.c; q++) {
                    res.channel(q).row(i)[j] = exp(res.channel(q).row(i)[j] - max);
                    sum += res.channel(q).row(i)[j];
                }

                for (int q = 0; q < res.c; q++) {
                    res.channel(q).row(i)[j] /= sum;
                }
            }
        }
    };

    std::vector<std::thread> threads;
    for (int c = 0; c < res.c; c++) {
        threads.push_back(std::thread(softmax_channel, c));
    }
    for (auto& t : threads) {
        t.join();
    }
}

std::vector<cv::Mat> MeterSegmentation::cut_roi_img(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    std::vector<cv::Mat> cut_images;
    cut_images.clear();
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
            obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Mat cut_image = image(obj.rect);
        cut_images.push_back(cut_image);

#ifdef VISUALIZE
        cv::imshow("sub_image", cut_image);
        cv::waitKey(0);
#endif // VISUALIZE
    }
    return cut_images;
}

cv::Mat MeterSegmentation::Process(const cv::Mat& input_image, cv::Mat& resize_image)
{
#ifdef VISUALIZE
    cv::imshow("input_image: ", input_image);
    cv::waitKey(0);
#endif

    // 将图像大小调整为目标大小
    float scale = ResizeImage(input_image, resize_image, cv::Size(DEEPLABV3P_TARGET_SIZE, DEEPLABV3P_TARGET_SIZE), cv::Scalar(128, 128, 128));

#ifdef VISUALIZE
    cv::imshow("resize_image", resize_image);
    cv::waitKey(0);
#endif

    // 运行分割模型
    ncnn::Mat res;
    run(resize_image, res);

#ifdef VISUALIZE
    std::cout << "Res shape: " << res.h << ", " << res.w << ", " << res.c << std::endl;
#endif

    // 代码后处理(针对仪表检测任务)
    cv::Mat mask = cv::Mat(DEEPLABV3P_TARGET_SIZE, DEEPLABV3P_TARGET_SIZE, CV_8UC1, cv::Scalar(0));

    const float* class0mask = res.channel(0);  // background
    const float* class1mask = res.channel(1);  // pointer
    const float* class2mask = res.channel(2);  // scale

    // 遍历每个像素，确定其类别
    for (int i = 0; i < DEEPLABV3P_TARGET_SIZE; i++) {
        for (int j = 0; j < DEEPLABV3P_TARGET_SIZE; j++) {
            int num = i * DEEPLABV3P_TARGET_SIZE + j;
            if ((class1mask[num] > class2mask[num]) && (class1mask[num] > class0mask[num])) {
                mask.at<uchar>(i, j) = 1;
            }
            else if ((class2mask[num] > class1mask[num]) && (class2mask[num] > class0mask[num])) {
                mask.at<uchar>(i, j) = 2;
            }
        }
    }

    res.release();

    return mask;
}



std::vector<cv::Mat> MeterSegmentation::preprocess(const std::vector<cv::Mat>& input_images, std::vector<cv::Mat>& resize_images)
{
    int meter_num = input_images.size();
    std::vector<cv::Mat> outputs;

    for (int i_num = 0; i_num < meter_num; i_num++)
    {
        cv::Mat input_image = input_images[i_num].clone();
        cv::Mat resize_image;

        std::cout << "current image shape: " << input_image.rows << ", " << input_image.cols << std::endl;

#ifdef VISUALIZE
        cv::imshow("input_image: ", input_image);
        cv::waitKey(0);
#endif

        float scale = ResizeImage(input_image, resize_image, cv::Size(DEEPLABV3P_TARGET_SIZE, DEEPLABV3P_TARGET_SIZE), cv::Scalar(128, 128, 128));
        std::cout << "current scale: " << scale << std::endl;

#ifdef VISUALIZE
        cv::imshow("resize_image", resize_image);
        cv::waitKey(0);
#endif

        ncnn::Mat res;
        run(resize_image, res);

#ifdef VISUALIZE
        std::cout << "Res shape: " << res.h << ", " << res.w << ", " << res.c << std::endl;
#endif

        cv::Mat mask(DEEPLABV3P_TARGET_SIZE, DEEPLABV3P_TARGET_SIZE, CV_8UC1, cv::Scalar(0));

        const float* class0mask = res.channel(0);
        const float* class1mask = res.channel(1);
        const float* class2mask = res.channel(2);

        for (int i = 0; i < DEEPLABV3P_TARGET_SIZE; i++)
        {
            for (int j = 0; j < DEEPLABV3P_TARGET_SIZE; j++)
            {
                int num = i * DEEPLABV3P_TARGET_SIZE + j;
                if (class1mask[num] > class2mask[num] && class1mask[num] > class0mask[num])
                {
                    mask.at<uchar>(i, j) = 1;
                }
                else if (class2mask[num] > class1mask[num] && class2mask[num] > class0mask[num])
                {
                    mask.at<uchar>(i, j) = 2;
                }
            }
        }

#ifdef VISUALIZE
        std::cout << "Res shape: " << res.h << ", " << res.w << ", " << res.c << std::endl;
        cv::imshow("mask_recover", mask * 100.0);
        cv::waitKey(50);
#endif

        outputs.push_back(mask);
        resize_images.push_back(resize_image);

        mask.release();
        input_image.release();
        resize_image.release();
        res.release();
    }

    return outputs;
}
