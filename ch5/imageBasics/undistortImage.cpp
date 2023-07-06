#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
string image_file = "../distorted.png";

int main(int argc, char **argv) {

    // distortion parameters
    double k1 = -0.28340811;
    double k2 = 0.07395907;
    double p1 = 0.00019359;
    double p2 = 1.76187114e-05;

    // intrinsic parameters
    double fx = 458.654;
    double fy = 457.926;
    double cx = 367.215;
    double cy = 248.375;

    // read image
    cv::Mat image = cv::imread(image_file, 0);  // gray
    int rows = image.rows;
    int cols = image.cols;

    // initialize undistort_image
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);

    // undistortion
    for (int v = 0; v < rows; v++) {
        for (int u = 0; u < cols; u++) {
            // calculate normalized coordinates
            double x_norm = (u - cx) / fx;
            double y_norm = (v - cy) / fy;
            double r = sqrt(x_norm * x_norm + y_norm * y_norm);

            // calculate distort x&y
            double x_distorted = x_norm * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x_norm * y_norm + p2 * (r * r + 2 * x_norm * x_norm);
            double y_distorted = y_norm * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y_norm * y_norm) + 2 * p2 * x_norm * y_norm;

            // calculate undistorted u&v
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;

            // assignment
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
            } else {
                image_undistort.at<uchar>(v, u) = 0;
            }
        }  
    }
    
    // draw undistort_image
    cv::imshow("distorted", image);
    cv::imshow("undistorted", image_undistort);
    cv::waitKey(0);

    return 0;
}