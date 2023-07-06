#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;


int main(int argc, char **argv) {

    // check
    if (argc != 3) {
        cout << "usage: LKByOpenCV.cpp LK1.png LK2.png" << endl;
        return 1;
    }

    // read image data
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

    // get img1's corner points
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);   // (max_corners, quality_level, min_distance)
    detector->detect(img1, kp1);

    // LK by OpenCV
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;       // if a point has been tracked, the status will be set to 1
    vector<float> error;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    cout << "use OpenCV solve LK cost: " << time_used.count() << " seconds." << endl;

    // show
    Mat img2_cv;
    cvtColor(img2, img2_cv, CV_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_cv, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_cv, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }
    imshow("tracked by OpenCV", img2_cv);
    waitKey(0);
    imwrite("/home/yynn/SLAM_14/ch8/opticalFlow/result/LK_OpenCV.png", img2_cv);

    return 0;
}