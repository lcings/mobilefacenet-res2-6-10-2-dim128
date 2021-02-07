#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;

cv::Mat norm_image(cv::Mat img)
{
    Mat img_float;
    img.convertTo(img_float, CV_32F);
    Mat img_norm = (img_float - 127.5) / 128;

    cv::Mat input_blob = cv::dnn::blobFromImage(img);
    float *p = (float*)input_blob.data;
    float *d = (float*)img_norm.data;
    int idx = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 112 * 112; j++) {
            //¶Ô±êtranspose(2,0,1)
            p[idx++] = d[3 * j + i];
        }
    }

    return input_blob;
}

int main()
{
    Mat image0 = cv::imread("crop_img1.bmp");
    Mat image1 = cv::imread("crop_img2.bmp");
    
    cv::Mat blob0 = norm_image(image0);
    cv::Mat blob1 = norm_image(image1);

    const std::vector<cv::String> targets_node{ "fc1" };
    std::vector< cv::Mat > targets_blobs0;

    dnn::Net net = cv::dnn::readNetFromCaffe("mobilefacenet-res2-6-10-2-dim128-opencv.prototxt", 
        "mobilefacenet-res2-6-10-2-dim128.caffemodel");
    
    net.setInput(blob0, "data");
    net.forward(targets_blobs0, targets_node);
    Mat out0 = targets_blobs0[0].clone();

    net.setInput(blob1, "data");
    net.forward(targets_blobs0, targets_node);
    Mat out1 = targets_blobs0[0].clone();

    float *feat0 = (float*)out0.data;
    float *feat1 = (float*)out1.data;

    for (int i = 0; i < 128; i++) {
        printf("%f ", feat0[i]);
    }
    
    putchar('\n');
    for (int i = 0; i < 128; i++) {
        printf("%f ", feat1[i]);
    }

    float sumxy = 0;
    float sumxx = 0;
    float sumyy = 0;
    for (int i = 0; i < 128; i++) {
        sumxy += feat0[i] * feat1[i];
        sumxx += feat0[i] * feat0[i];
        sumyy += feat1[i] * feat1[i];
    }

    float score = sumxy / sqrt(sumxx * sumyy);

    std::cout << "\nSimilarity score: " << score << "\n";
    system("pause");
    return 0;
}